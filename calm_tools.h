/**
 * calm_tools.h — Calm Tool/Function Calling
 *
 * Minimal implementation of Qwen2.5-compatible tool calling:
 *   - JSON parsing for tool definitions and tool calls
 *   - Qwen2.5 chat template formatting with tools
 *   - <|tool_call|> detection and parsing
 *   - Multi-step tool call loop
 *
 * Usage:
 *   CalmToolDefinitions tools = {0};
 *   ct_tools_add(&tools, "get_weather", "Get weather",
 *       "{\"type\":\"object\",\"properties\":{...}}");
 *
 *   char system_buf[4096];
 *   ct_tools_format_system(&tools, system_buf, sizeof(system_buf));
 *
 *   // After generation, parse tool calls:
 *   CalmToolCall calls[4];
 *   int n = ct_tools_parse(model_output, calls, 4);
 *   for (int i = 0; i < n; i++) {
 *       CalmToolResult res;
 *       ct_tools_execute(&calls[i], &res);
 *       // ... format result and continue generation
 *   }
 */
#ifndef CALM_TOOLS_H
#define CALM_TOOLS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ─── */
#define CT_TOOLS_MAX 16          /* max number of tool definitions */
#define CT_TOOL_NAME_MAX 64
#define CT_TOOL_DESC_MAX 512
#define CT_TOOL_PARAMS_MAX 2048  /* JSON schema for parameters */
#define CT_TOOL_ARGS_MAX 4096    /* JSON arguments in tool call */
#define CT_TOOL_RESULT_MAX 8192  /* JSON result from tool execution */
#define CT_TOOL_CALLS_MAX 8      /* max parallel tool calls per step */

/* ─── Tool definition (user-provided) ─── */
typedef struct {
    char name[CT_TOOL_NAME_MAX];
    char description[CT_TOOL_DESC_MAX];
    char parameters[CT_TOOL_PARAMS_MAX]; /* JSON schema string */
} CalmToolFunction;

typedef struct {
    int count;
    CalmToolFunction functions[CT_TOOLS_MAX];
} CalmToolDefinitions;

/* ─── Tool call (parsed from model output) ─── */
typedef struct {
    char name[CT_TOOL_NAME_MAX];
    char arguments[CT_TOOL_ARGS_MAX]; /* JSON string */
} CalmToolCall;

/* ─── Tool execution result ─── */
typedef struct {
    char content[CT_TOOL_RESULT_MAX]; /* JSON string */
} CalmToolResult;

/* ─── Tool call step result ─── */
typedef struct {
    int  n_calls;
    CalmToolCall calls[CT_TOOL_CALLS_MAX];
} CalmToolCallResult;

/* ─── API ─── */

/* Add a tool definition. Returns 0 on success, -1 on error. */
int ct_tools_add(CalmToolDefinitions* tools,
                 const char* name, const char* description,
                 const char* parameters_json);

/* Format tool definitions into Qwen2.5 system prompt suffix.
 * Appends JSON tool definitions after the system message.
 * Returns bytes written. */
int ct_tools_format_system(const CalmToolDefinitions* tools,
                           char* buf, size_t buf_size);

/* Parse model output for <|tool_call|> blocks.
 * Returns number of tool calls found, or 0 if none. */
int ct_tools_parse(const char* output, CalmToolCall* calls, int max_calls);

/* Execute a tool call using built-in stub functions.
 * For MVP: returns a formatted echo of the call.
 * Future: user-provided callback dispatch.
 * Returns 0 on success. */
int ct_tools_execute(const CalmToolCall* call, CalmToolResult* result);

/* Format a tool result into a Qwen2.5 tool response chunk.
 * Returns bytes written. */
int ct_tools_format_result(const CalmToolCall* call,
                           const CalmToolResult* result,
                           char* buf, size_t buf_size);

/* Check if output contains a tool call (quick check without parsing). */
int ct_tools_has_call(const char* output);

/* ─── Minimal JSON helpers ─── */

/* Extract string value by key from a JSON object.
 * Returns pointer to value (in-place, null-terminated) or NULL.
 * Only handles flat string values. */
const char* ct_json_get_string(const char* json, const char* key);

/* Extract integer value by key. Returns 0 on missing/invalid. */
int ct_json_get_int(const char* json, const char* key, int default_val);

/* Extract float value by key. Returns default_val on missing/invalid. */
float ct_json_get_float(const char* json, const char* key, float default_val);

/* Extract a value (any type) by key. Returns pointer to value start. */
const char* ct_json_get_value(const char* json, const char* key);

/* Skip one JSON value, return pointer after it. */
const char* ct_json_skip_value(const char* p);

/* Parse a JSON string from p, write unescaped to out, return end. */
const char* ct_json_parse_string(const char* p, char* out, size_t max);

/* Find tool definitions in the system prompt (for parsing tool use). */
int ct_tools_parse_definitions(const char* json,
                               CalmToolDefinitions* tools);

#ifdef __cplusplus
}
#endif

#endif /* CALM_TOOLS_H */
