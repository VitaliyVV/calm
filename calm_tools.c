/**
 * calm_tools.c — Calm Tool/Function Calling Implementation
 *
 * Implements Qwen2.5-compatible tool calling with minimal JSON parsing.
 *
 * The Qwen2.5 tool calling protocol:
 *   1. System prompt includes tool definitions as JSON
 *   2. Model generates: some text <|tool_call|>{"name":"...","arguments":{...}}
 *   3. Host executes the function and returns: <|im_start|>tool\n<|tool_call|>\n{result}\n<|im_end|>\n<|im_start|>assistant\n
 *   4. Model generates final text response
 */
#if defined(_MSC_VER)
/* MSVC deprecates strncpy (C4996); the bounded copies below are intentional */
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "calm_tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

/* ═══════════════════════════════════════════════════════════════
 * Internal: minimal JSON parser (recursive descent)
 * ═══════════════════════════════════════════════════════════════ */

/* Skip whitespace */
static const char* json_skip_ws(const char* p) {
    while (*p && (unsigned char)*p <= ' ' && *p != '\0') p++;
    return p;
}

/* Check if character starts a JSON value */
static int json_is_value_start(char c) {
    return c == '"' || c == '{' || c == '[' || c == 't' || c == 'f' ||
           c == 'n' || c == '-' || (c >= '0' && c <= '9');
}

/* Parse a JSON string from p, write unescaped to out, return end pointer.
 * Handles \", \\, \/, \b, \f, \n, \r, \t, \uXXXX. */
const char* ct_json_parse_string(const char* p, char* out, size_t max) {
    if (!p || *p != '"') return NULL;
    p++; /* skip opening quote */
    size_t i = 0;
    while (*p && *p != '"' && i < max - 1) {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case '"':  out[i++] = '"'; break;
                case '\\': out[i++] = '\\'; break;
                case '/':  out[i++] = '/'; break;
                case 'b':  out[i++] = '\b'; break;
                case 'f':  out[i++] = '\f'; break;
                case 'n':  out[i++] = '\n'; break;
                case 'r':  out[i++] = '\r'; break;
                case 't':  out[i++] = '\t'; break;
                case 'u': {
                    /* Simplified: skip \uXXXX, store replacement char */
                    if (p[1] && p[2] && p[3] && p[4]) {
                        /* Skip 4 hex digits */
                        p += 4;
                        out[i++] = '?';
                    }
                    break;
                }
                default: out[i++] = *p; break;
            }
            if (*p) p++;
        } else {
            out[i++] = *p++;
        }
    }
    out[i] = '\0';
    if (*p == '"') p++; /* skip closing quote */
    return p;
}

/* Skip one JSON value, return pointer after it */
const char* ct_json_skip_value(const char* p) {
    if (!p) return NULL;
    p = json_skip_ws(p);
    if (!*p) return p;

    switch (*p) {
        case '"': {
            /* Skip string */
            p++;
            while (*p) {
                if (*p == '\\') { if (p[1]) p += 2; else break; }
                else if (*p == '"') { p++; break; }
                else p++;
            }
            return p;
        }
        case '{':
        case '[': {
            char brace = *p;
            char close = (brace == '{') ? '}' : ']';
            p++;
            int depth = 1;
            while (*p && depth > 0) {
                if (*p == '"') {
                    /* Skip string */
                    p++;
                    while (*p) {
                        if (*p == '\\') { if (p[1]) p += 2; else break; }
                        else if (*p == '"') { p++; break; }
                        else p++;
                    }
                } else {
                    if (*p == brace) depth++;
                    if (*p == close) depth--;
                    if (depth > 0) p++;
                }
            }
            if (*p) p++;
            return p;
        }
        case 't': /* true */  return (*p == 't') ? p + 4 : p;
        case 'f': /* false */ return (*p == 'f') ? p + 5 : p;
        case 'n': /* null */  return (*p == 'n') ? p + 4 : p;
        default: {
            /* Number */
            if (*p == '-' || (*p >= '0' && *p <= '9')) {
                p++;
                while (*p && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-'))
                    p++;
            }
            return p;
        }
    }
}

/* Get pointer to the value of a key in a JSON object.
 * Returns pointer to the start of the value, or NULL.
 * The returned pointer is NOT null-terminated — use the next step to parse it. */
const char* ct_json_get_value(const char* json, const char* key) {
    if (!json || !key) return NULL;
    size_t klen = strlen(key);

    const char* p = json_skip_ws(json);
    if (*p != '{') return NULL;
    p++; /* skip { */

    while (*p) {
        p = json_skip_ws(p);
        if (*p == '}') return NULL;
        if (*p == ',') p++;

        p = json_skip_ws(p);
        if (*p != '"') return NULL;

        /* Parse key */
        p++; /* skip " */
        const char* kstart = p;
        while (*p && *p != '"') {
            if (*p == '\\') { if (p[1]) p += 2; else break; }
            else p++;
        }
        size_t found_len = (size_t)(p - kstart);
        if (*p == '"') p++; /* skip " */

        p = json_skip_ws(p);
        if (*p != ':') return NULL;
        p++; /* skip : */
        p = json_skip_ws(p);

        if (found_len == klen && memcmp(kstart, key, klen) == 0) {
            return p; /* pointer to value start */
        }

        /* Skip value */
        p = ct_json_skip_value(p);
        if (!p) return NULL;
    }
    return NULL;
}

/* Extract string value by key from a JSON object.
 * Writes null-terminated string to a static buffer.
 * Returns pointer to buffer, or NULL if not found.
 * NOTE: uses a small static buffer — only for simple string extraction. */
static char json_str_buf[CT_TOOL_PARAMS_MAX];

const char* ct_json_get_string(const char* json, const char* key) {
    const char* val = ct_json_get_value(json, key);
    if (!val) return NULL;
    if (*val != '"') return NULL;
    if (!ct_json_parse_string(val, json_str_buf, sizeof(json_str_buf)))
        return NULL;
    return json_str_buf;
}

/* ═══════════════════════════════════════════════════════════════
 * Tool definitions from JSON array
 * ═══════════════════════════════════════════════════════════════ */

int ct_tools_parse_definitions(const char* json, CalmToolDefinitions* tools) {
    if (!json || !tools) return -1;
    tools->count = 0;

    const char* p = json_skip_ws(json);
    if (*p != '[') return -1;
    p++; /* skip [ */

    while (*p && tools->count < CT_TOOLS_MAX) {
        p = json_skip_ws(p);
        if (*p == ']') break;
        if (*p == ',') { p++; continue; }
        if (*p != '{') break;

        /* Find the entire tool object */
        const char* obj_start = p;
        p = ct_json_skip_value(p);
        if (!p) break;
        /* Temporarily null-terminate the object for parsing */
        /* We'll work with what we have */

        /* Look for "function" key (OpenAI format: {type:"function",function:{...}}) */
        const char* func_val = ct_json_get_value(obj_start, "function");
        if (!func_val) {
            /* Direct format: {name:"...",description:"...",parameters:{...}} */
            func_val = obj_start;
        }

        /* Make a local copy for safe parsing */
        char obj_copy[CT_TOOL_PARAMS_MAX + 256];
        size_t obj_len = (size_t)(p - obj_start);
        if (obj_len >= sizeof(obj_copy) - 1) obj_len = sizeof(obj_copy) - 1;
        memcpy(obj_copy, obj_start, obj_len);
        obj_copy[obj_len] = '\0';

        CalmToolFunction* f = &tools->functions[tools->count];
        memset(f, 0, sizeof(CalmToolFunction));

        /* Extract name, description, parameters */
        const char* name_val = ct_json_get_string(obj_copy, "name");
        if (name_val) {
            strncpy(f->name, name_val, sizeof(f->name) - 1);
        } else {
            /* Try nested function.name */
            const char* fc_val = ct_json_get_value(obj_copy, "function");
            if (fc_val) {
                /* Make a sub-copy of the function object */
                const char* fc_end = ct_json_skip_value(fc_val);
                size_t fc_len = fc_end ? (size_t)(fc_end - fc_val) : strlen(fc_val);
                char fc_copy[CT_TOOL_PARAMS_MAX];
                if (fc_len >= sizeof(fc_copy)) fc_len = sizeof(fc_copy) - 1;
                memcpy(fc_copy, fc_val, fc_len);
                fc_copy[fc_len] = '\0';
                name_val = ct_json_get_string(fc_copy, "name");
                if (name_val) strncpy(f->name, name_val, sizeof(f->name) - 1);
            }
        }

        const char* desc_val = ct_json_get_string(obj_copy, "description");
        if (!desc_val) {
            const char* fc_val = ct_json_get_value(obj_copy, "function");
            if (fc_val) {
                const char* fc_end = ct_json_skip_value(fc_val);
                size_t fc_len = fc_end ? (size_t)(fc_end - fc_val) : strlen(fc_val);
                char fc_copy[CT_TOOL_PARAMS_MAX];
                if (fc_len >= sizeof(fc_copy)) fc_len = sizeof(fc_copy) - 1;
                memcpy(fc_copy, fc_val, fc_len);
                fc_copy[fc_len] = '\0';
                desc_val = ct_json_get_string(fc_copy, "description");
                if (desc_val) strncpy(f->description, desc_val, sizeof(f->description) - 1);
            }
        } else {
            strncpy(f->description, desc_val, sizeof(f->description) - 1);
        }

        /* Extract parameters JSON (store raw JSON object) */
        const char* params_val = ct_json_get_value(obj_copy, "parameters");
        if (!params_val) {
            const char* fc_val = ct_json_get_value(obj_copy, "function");
            if (fc_val) {
                const char* fc_end = ct_json_skip_value(fc_val);
                size_t fc_len = fc_end ? (size_t)(fc_end - fc_val) : strlen(fc_val);
                char fc_copy[CT_TOOL_PARAMS_MAX];
                if (fc_len >= sizeof(fc_copy)) fc_len = sizeof(fc_copy) - 1;
                memcpy(fc_copy, fc_val, fc_len);
                fc_copy[fc_len] = '\0';
                params_val = ct_json_get_value(fc_copy, "parameters");
            }
        }
        if (params_val) {
            const char* params_end = ct_json_skip_value(params_val);
            size_t params_len = params_end ? (size_t)(params_end - params_val) : strlen(params_val);
            if (params_len >= sizeof(f->parameters)) params_len = sizeof(f->parameters) - 1;
            memcpy(f->parameters, params_val, params_len);
            f->parameters[params_len] = '\0';
        }

        if (f->name[0]) {
            tools->count++;
        }
    }

    return tools->count;
}

/* ═══════════════════════════════════════════════════════════════
 * Add tool definition
 * ═══════════════════════════════════════════════════════════════ */

int ct_tools_add(CalmToolDefinitions* tools,
                 const char* name, const char* description,
                 const char* parameters_json) {
    if (!tools || !name || tools->count >= CT_TOOLS_MAX) return -1;

    CalmToolFunction* f = &tools->functions[tools->count];
    memset(f, 0, sizeof(CalmToolFunction));

    strncpy(f->name, name, sizeof(f->name) - 1);
    if (description)
        strncpy(f->description, description, sizeof(f->description) - 1);
    if (parameters_json)
        strncpy(f->parameters, parameters_json, sizeof(f->parameters) - 1);

    tools->count++;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Format tools into Qwen2.5 system prompt
 *
 * Qwen2.5 format:
 *   You have access to the following functions. Use them if required:
 *
 *   {
 *     "type": "function",
 *     "function": {
 *       "name": "<name>",
 *       "description": "<description>",
 *       "parameters": { ... }
 *     }
 *   }
 * ═══════════════════════════════════════════════════════════════ */

int ct_tools_format_system(const CalmToolDefinitions* tools,
                           char* buf, size_t buf_size) {
    if (!tools || tools->count == 0 || !buf || buf_size == 0) return 0;

    size_t pos = 0;
    int n = snprintf(buf + pos, buf_size - pos,
        "\n\nYou have access to the following functions. Use them if required:\n");
    if (n > 0 && (size_t)n < buf_size - pos) pos += (size_t)n;

    for (int i = 0; i < tools->count && pos < buf_size; i++) {
        const CalmToolFunction* f = &tools->functions[i];
        n = snprintf(buf + pos, buf_size - pos,
            "\n{\n"
            "  \"type\": \"function\",\n"
            "  \"function\": {\n"
            "    \"name\": \"%s\",\n"
            "    \"description\": \"%s\",\n"
            "    \"parameters\": %s\n"
            "  }\n"
            "}",
            f->name, f->description, f->parameters);
        if (n > 0 && (size_t)n < buf_size - pos) pos += (size_t)n;
    }

    n = snprintf(buf + pos, buf_size - pos, "\n");
    if (n > 0 && (size_t)n < buf_size - pos) pos += (size_t)n;

    return (int)pos;
}

/* ═══════════════════════════════════════════════════════════════
 * Tool call detection and parsing
 *
 * Model output format (Qwen2.5):
 *   <|tool_call|>\n{"name":"...","arguments":{...}}\n<|im_end|>
 *
 * Or inline:
 *   Let me check the weather. <|tool_call|>{"name":"get_weather","arguments":{"location":"Moscow"}}
 * ═══════════════════════════════════════════════════════════════ */

int ct_tools_has_call(const char* output) {
    return output && strstr(output, "<|tool_call|>") != NULL;
}

int ct_tools_parse(const char* output, CalmToolCall* calls, int max_calls) {
    if (!output || !calls || max_calls <= 0) return 0;

    int n = 0;
    const char* p = output;
    const char* tool_call_tag = "<|tool_call|>";

    while (*p && n < max_calls) {
        /* Find next <|tool_call|> tag */
        const char* tag = strstr(p, tool_call_tag);
        if (!tag) break;

        const char* json_start = tag + strlen(tool_call_tag);
        /* Skip whitespace/newlines after tag */
        while (*json_start && (unsigned char)*json_start <= ' ' && *json_start != '\0')
            json_start++;

        if (*json_start != '{') {
            p = json_start;
            continue;
        }

        /* Parse the JSON object */
        char json_copy[CT_TOOL_ARGS_MAX];
        size_t jpos = 0;
        const char* jp = json_start;
        int depth = 0;
        int in_string = 0;

        while (*jp && jpos < sizeof(json_copy) - 1) {
            if (*jp == '"' && (jp == json_start || *(jp-1) != '\\')) {
                in_string = !in_string;
            }
            if (!in_string) {
                if (*jp == '{') depth++;
                if (*jp == '}') depth--;
                if (depth == 0 && jp > json_start && *jp == '}') {
                    json_copy[jpos++] = *jp;
                    jp++;
                    break;
                }
            }
            json_copy[jpos++] = *jp++;
        }
        json_copy[jpos] = '\0';

        if (jpos == 0) { p = json_start + 1; continue; }

        /* Extract name and arguments */
        CalmToolCall* call = &calls[n];
        memset(call, 0, sizeof(CalmToolCall));

        const char* name_val = ct_json_get_string(json_copy, "name");
        if (!name_val) {
            /* Maybe the model skipped the wrapper: just { ... } with function call */
            /* Try to find "function" key */
            const char* fc = ct_json_get_value(json_copy, "function");
            if (fc) {
                /* Skip value to get a null-terminated copy */
                const char* fc_end = ct_json_skip_value(fc);
                size_t fc_len = fc_end ? (size_t)(fc_end - fc) : strlen(fc);
                char fc_copy[CT_TOOL_ARGS_MAX];
                if (fc_len >= sizeof(fc_copy)) fc_len = sizeof(fc_copy) - 1;
                memcpy(fc_copy, fc, fc_len);
                fc_copy[fc_len] = '\0';
                name_val = ct_json_get_string(fc_copy, "name");
            }
        }

        if (name_val) {
            strncpy(call->name, name_val, sizeof(call->name) - 1);

            /* Extract arguments as raw JSON */
            const char* args_val = ct_json_get_value(json_copy, "arguments");
            if (args_val) {
                const char* args_end = ct_json_skip_value(args_val);
                size_t args_len = args_end ? (size_t)(args_end - args_val) : strlen(args_val);
                if (args_len >= sizeof(call->arguments)) args_len = sizeof(call->arguments) - 1;
                memcpy(call->arguments, args_val, args_len);
                call->arguments[args_len] = '\0';
            }

            n++;
        }

        p = jp;
    }

    return n;
}

/* ═══════════════════════════════════════════════════════════════
 * Tool execution (MVP: built-in stub functions)
 * ═══════════════════════════════════════════════════════════════ */

/* Built-in tool implementations */
static int exec_get_current_time(CalmToolCall* call, CalmToolResult* result) {
    (void)call;
    /* arguments: {} — no params */
    time_t now = time(NULL);
    struct tm* tm = localtime(&now);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
    snprintf(result->content, sizeof(result->content),
             "{\"result\": \"%s\"}", buf);
    return 0;
}

static int exec_get_weather(CalmToolCall* call, CalmToolResult* result) {
    /* Extract location from arguments */
    const char* loc = ct_json_get_string(call->arguments, "location");
    const char* unit = ct_json_get_string(call->arguments, "unit");
    if (!loc) loc = "unknown";
    if (!unit) unit = "celsius";

    snprintf(result->content, sizeof(result->content),
             "{\"result\": \"Weather in %s: sunny, 22°%s\"}", loc,
             (strcmp(unit, "fahrenheit") == 0) ? "F" : "C");
    return 0;
}

static int exec_search(CalmToolCall* call, CalmToolResult* result) {
    const char* query = ct_json_get_string(call->arguments, "query");
    if (!query) query = "";
    snprintf(result->content, sizeof(result->content),
             "{\"result\": \"Search results for '%s': (simulated) no results found.\"}", query);
    return 0;
}

static int exec_calculator(CalmToolCall* call, CalmToolResult* result) {
    const char* expr = ct_json_get_string(call->arguments, "expression");
    if (!expr) expr = "";
    /* Simple expression evaluation not implemented yet */
    snprintf(result->content, sizeof(result->content),
             "{\"result\": \"Calculator: %s (computation not available)\"}", expr);
    return 0;
}

typedef struct {
    const char* name;
    int (*func)(CalmToolCall*, CalmToolResult*);
} BuiltinTool;

static const BuiltinTool builtin_tools[] = {
    {"get_current_time", exec_get_current_time},
    {"get_current_weather", exec_get_weather},
    {"get_weather", exec_get_weather},
    {"search", exec_search},
    {"web_search", exec_search},
    {"calculator", exec_calculator},
    {"calculate", exec_calculator},
    {NULL, NULL}
};

int ct_tools_execute(const CalmToolCall* call, CalmToolResult* result) {
    if (!call || !result) return -1;

    /* Make a mutable copy for the exec functions (which modify it) */
    CalmToolCall mutable_call;
    memcpy(&mutable_call, call, sizeof(mutable_call));

    /* Try to find a builtin handler */
    for (int i = 0; builtin_tools[i].name; i++) {
        if (strcmp(call->name, builtin_tools[i].name) == 0) {
            return builtin_tools[i].func(&mutable_call, result);
        }
    }

    /* Unknown tool: return a descriptive error */
    snprintf(result->content, sizeof(result->content),
             "{\"error\": \"Unknown function '%s'. Available functions: get_current_time, get_weather, search, calculator\"}",
             call->name);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Format tool result for continuation
 *
 * Qwen2.5 tool response format:
 *   <|im_start|>tool
 *   <|tool_call|>
 *   {"result": ...}
 *   <|im_end|>
 *   <|im_start|>assistant
 * ═══════════════════════════════════════════════════════════════ */

int ct_tools_format_result(const CalmToolCall* call,
                           const CalmToolResult* result,
                           char* buf, size_t buf_size) {
    (void)call;
    if (!buf || buf_size == 0) return 0;
    return snprintf(buf, buf_size,
        "<|im_start|>tool\n<|tool_call|>\n%s\n<|im_end|>\n<|im_start|>assistant\n",
        result ? result->content : "{\"error\": \"no result\"}");
}

/* ─── JSON numeric helpers ─── */
int ct_json_get_int(const char* json, const char* key, int default_val) {
    const char* val = ct_json_get_value(json, key);
    if (!val) return default_val;
    char* end = NULL;
    long n = strtol(val, &end, 10);
    if (end == val) return default_val;
    return (int)n;
}

float ct_json_get_float(const char* json, const char* key, float default_val) {
    const char* val = ct_json_get_value(json, key);
    if (!val) return default_val;
    char* end = NULL;
    double f = strtod(val, &end);
    if (end == val) return default_val;
    return (float)f;
}
