#pragma once

#include <cstddef>
#include <utility>
#include <mc_runtime.h>
#include "helper_maca.h"

template <class T>
struct MacaAllocator {
    using value_type = T;

    T *allocate(size_t size) {
        T *ptr = nullptr;
        checkMacaErrors(mcMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0) {
        checkMacaErrors(mcFree(ptr));
    }

    template <class ...Args>
    void construct(T *p, Args &&...args) {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))
            ::new((void *)p) T(std::forward<Args>(args)...);
    }

    constexpr bool operator==(MacaAllocator<T> const &other) const {
        return this == &other;
    }
};
