#ifndef CALM_SERVER_H
#define CALM_SERVER_H

#include <stddef.h>

/* ─── HTTP Request ─── */
typedef struct {
    char method[16];
    char path[1024];
    char body[65536];
    size_t body_len;
    int keep_alive;          /* 1 = Connection: keep-alive */
} ct_http_request;

/* ─── Route handler ───
 * Fills response_body, sets out_status_code and out_content_type.
 * Returns 0 on success, -1 on error (→ 500).
 * `body` is the raw request body. path and method are provided separately.
 */
typedef int (*ct_route_handler)(const char* path, const char* method,
                                 const char* body,
                                 char* response_body, size_t response_size,
                                 int* out_status_code,
                                 const char** out_content_type,
                                 void* user_data);

/* ─── Start HTTP server ───
 * Blocks until fatal error or SIGINT.
 * Port 0 = default 8080.
 */
int ct_server_start(int port, ct_route_handler handler, void* user_data);

/* ─── SSE streaming handler ───
 * Called from server when stream=true in request body.
 * Writes SSE events to fd, then writes [DONE]. Caller closes fd.
 * Returns 0 on success, -1 on error.
 * user_data must be the same as passed to ct_server_start.
 */
int calm_serve_sse(int fd, const char* body, void* user_data);

#endif /* CALM_SERVER_H */
