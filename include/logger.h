#ifndef IDRAK_LOGGER
#define IDRAK_LOGGER

#include <stdio.h>

#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define RESET "\033[0m"

#define LOG_INFO(msg, ...) printf(GREEN "[INFO] " RESET msg "\n", ##__VA_ARGS__)
#define LOG_WARN(msg, ...) printf(YELLOW "[WARNING] " RESET msg "\n", ##__VA_ARGS__)
#define LOG_ERROR(msg, ...) printf(RED "[ERROR] " RESET msg "\n", ##__VA_ARGS__)

#ifdef DEBUG
#define LOG_DEBUG(fmt, ...) printf(BLUE " [DEBUG] " RESET msg "\n", ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...)
#endif

#endif
