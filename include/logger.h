#ifndef AXON_LOGGER
#define AXON_LOGGER

#include <stdio.h>
#include <stdlib.h>

#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define RESET "\033[0m"

static int _idrak_debug_enabled = -1;

static inline int idrak_is_debug_enabled(void)
{
    if (_idrak_debug_enabled == -1)
    {
        const char* env = getenv("DEBUG");
        _idrak_debug_enabled = (env && env[0] == '1') ? 1 : 0;
    }
    return _idrak_debug_enabled;
}

#define LOG_INFO(msg, ...)                                                                         \
    do                                                                                             \
    {                                                                                              \
        if (idrak_is_debug_enabled()) printf(GREEN "[INFO] " RESET msg "\n", ##__VA_ARGS__);       \
    } while (0)

#define LOG_WARN(msg, ...) printf(YELLOW "[WARNING] " RESET msg "\n", ##__VA_ARGS__)
#define LOG_ERROR(msg, ...) printf(RED "[ERROR] " RESET msg "\n", ##__VA_ARGS__)

#endif
