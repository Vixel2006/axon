#include "tensor_cpu.h"
#include "logger.h"
#include "tensor.h"
#include <stdlib.h>
#include <string.h>

Storage* smalloc_cpu(float* data, int size, Device* device)
{
    if (!device)
    {
        LOG_ERROR("smalloc_cpu requires a non-NULL device.");
        return NULL;
    }

    Storage* s = malloc(sizeof(Storage));

    if (!s)
    {
        LOG_ERROR("Failed to allocate Storage");
        return NULL;
    }

    s->size = size;

    s->data = malloc(sizeof(float) * size);

    if (!s->data)
    {
        LOG_ERROR("Failed to allocate Storage data on cpu.");
        free(s);
        return NULL;
    }

    if (data != NULL)
    {
        for (int i = 0; i < size; ++i)
            s->data[i] = data[i];
    }
    else
    {
        for (int i = 0; i < size; ++i)
            s->data[i] = 0.0f;
    }

    s->counter = 1;

    LOG_INFO("Storage allocated at %p with size=%d", s, size);
    return s;
}

void sfree_cpu(Storage* s, Device* device)
{
    if (!s) return;
    s->counter--;
    LOG_INFO("sfree_cpu called, counter=%d for Storage %p", s->counter, s);
    if (s->counter <= 0)
    {
        if (s->data)
        {
            LOG_INFO("Storage data freed at %p", s->data);
            SAFE_FREE(&s->data, free);
        }
        SAFE_FREE(&s, free);
    }
}
