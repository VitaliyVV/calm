/**
 * calm_server.c — Minimal HTTP server (POSIX sockets, zero dependencies)
 *
 * Serves OpenAI-compatible /v1/completions endpoint.
 * Thread-per-request model; single model instance shared via user_data.
 */

#include "calm_server.h"

#include <ctype.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>

#define REQUEST_BUF_SIZE (128 * 1024)
#define RESPONSE_BUF_SIZE (128 * 1024)
#define MAX_HEADERS 64
#define MAX_HEADER_LEN 4096

/* Running flag for graceful shutdown */
static volatile int server_running = 1;

/* ─── Signal handler ─── */
static void sigint_handler(int sig) {
    (void)sig;
    server_running = 0;
}

/* ─── strncasecmp for platforms missing it ─── */
static int strncasecmp_s(const char* a, const char* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int ca = (unsigned char)tolower((unsigned char)a[i]);
        int cb = (unsigned char)tolower((unsigned char)b[i]);
        if (ca != cb) return ca - cb;
        if (a[i] == '\0') return 0;
    }
    return 0;
}

/* ─── Client job ─── */
typedef struct {
    int fd;
    ct_route_handler handler;
    void* user_data;
} client_job;

/* ─── Parse HTTP request from raw bytes ─── */
static int parse_http_request(const char* raw, size_t raw_len,
                               ct_http_request* req, size_t* body_offset) {
    memset(req, 0, sizeof(*req));
    *body_offset = 0;

    /* Copy to writable buffer */
    char buf[REQUEST_BUF_SIZE];
    size_t copy_len = raw_len < sizeof(buf) - 1 ? raw_len : sizeof(buf) - 1;
    memcpy(buf, raw, copy_len);
    buf[copy_len] = '\0';

    /* Parse request line: METHOD PATH HTTP/1.1 */
    char* line = buf;
    char* end = strstr(line, "\r\n");
    if (!end) return -1;
    *end = '\0';

    char http_ver[16];
    if (sscanf(line, "%15s %1023s %15s", req->method, req->path, http_ver) < 2)
        return -1;

    /* Parse headers */
    line = end + 2;
    int content_length = 0;
    while (1) {
        end = strstr(line, "\r\n");
        if (!end) break;
        *end = '\0';
        if (line[0] == '\0') {
            /* End of headers */
            *body_offset = (size_t)(end + 2 - buf);
            break;
        }
        if (strncasecmp_s(line, "Content-Length:", 15) == 0) {
            const char* val = line + 15;
            while (*val == ' ') val++;
            content_length = atoi(val);
        }
        line = end + 2;
    }

    /* Copy body */
    if (content_length > 0 && *body_offset + (size_t)content_length <= raw_len) {
        size_t body_max = sizeof(req->body) - 1;
        req->body_len = (size_t)content_length < body_max ? (size_t)content_length : body_max;
        memcpy(req->body, raw + *body_offset, req->body_len);
        req->body[req->body_len] = '\0';
    }

    return 0;
}

/* ─── Build HTTP response string ─── */
static int build_http_response(char* buf, size_t buf_size,
                                int status_code, const char* content_type,
                                const char* body) {
    return snprintf(buf, buf_size,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n"
        "%s",
        status_code,
        status_code == 200 ? "OK" :
        status_code == 400 ? "Bad Request" :
        status_code == 404 ? "Not Found" :
        status_code == 500 ? "Internal Server Error" : "Unknown",
        content_type ? content_type : "text/plain",
        body ? strlen(body) : 0,
        body ? body : "");
}

/* ─── Thread: handle one client ─── */
static void* client_thread(void* arg) {
    client_job* job = (client_job*)arg;
    int fd = job->fd;
    ct_route_handler handler = job->handler;
    void* user_data = job->user_data;
    free(job);

    char raw[REQUEST_BUF_SIZE];
    size_t total = 0;

    /* Read request */
    while (total < sizeof(raw) - 1) {
        int n = (int)read(fd, raw + total, sizeof(raw) - 1 - total);
        if (n <= 0) break;
        total += (size_t)n;
        raw[total] = '\0';

        /* Check if we have full headers */
        char* hdr_end = strstr(raw, "\r\n\r\n");
        if (hdr_end) {
            /* Parse Content-Length */
            int body_len = 0;
            char* cl = strstr(raw, "Content-Length:");
            if (!cl) cl = strstr(raw, "content-length:");
            if (cl) {
                cl += 15;
                while (*cl == ' ') cl++;
                body_len = atoi(cl);
            }
            size_t header_end_offset = (size_t)(hdr_end + 4 - raw);
            if (header_end_offset + (size_t)body_len <= total) {
                break; /* Full request received */
            }
        }
    }

    if (total == 0) {
        close(fd);
        return NULL;
    }

    /* Parse HTTP request */
    ct_http_request http_req;
    size_t body_offset = 0;
    if (parse_http_request(raw, total, &http_req, &body_offset) != 0) {
        char resp[4096];
        size_t n = build_http_response(resp, sizeof(resp), 400, "text/plain", "Bad Request");
        write(fd, resp, n);
        close(fd);
        return NULL;
    }

    /* Call the route handler */
    char response_body[RESPONSE_BUF_SIZE];
    int status_code = 500;
    const char* content_type = "application/json";
    int ret = handler(http_req.path, http_req.method,
                       http_req.body_len > 0 ? http_req.body : NULL,
                       response_body, sizeof(response_body),
                       &status_code, &content_type, user_data);

    if (ret != 0 && status_code == 200) status_code = 500;

    char resp[RESPONSE_BUF_SIZE + 4096];
    size_t n = build_http_response(resp, sizeof(resp),
                                    status_code, content_type, response_body);
    write(fd, resp, n);
    close(fd);
    return NULL;
}

/* ─── Start server (blocks) ─── */
int ct_server_start(int port, ct_route_handler handler, void* user_data) {
    if (port <= 0 || port > 65535) port = 8080;

    /* Register SIGINT handler */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sigint_handler;
    sigaction(SIGINT, &sa, NULL);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return -1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return -1;
    }

    if (listen(server_fd, 5) < 0) {
        perror("listen");
        close(server_fd);
        return -1;
    }

    char port_str[32];
    snprintf(port_str, sizeof(port_str), "%d", port);

    /* Write port to stdout for scripts */
    printf("calm: server listening on 0.0.0.0:%s\n", port_str);
    printf("calm: POST /v1/completions\n");
    fflush(stdout);

    while (server_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (server_running) perror("accept");
            continue;
        }

        client_job* job = (client_job*)malloc(sizeof(client_job));
        if (!job) {
            close(client_fd);
            continue;
        }
        job->fd = client_fd;
        job->handler = handler;
        job->user_data = user_data;

        pthread_t thread;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

        if (pthread_create(&thread, &attr, client_thread, job) != 0) {
            free(job);
            close(client_fd);
        }
        pthread_attr_destroy(&attr);
    }

    close(server_fd);
    printf("calm: server stopped\n");
    return 0;
}
