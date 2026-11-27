/*
 * This file is part of eigen-mcu, an MCU-targeted fork of Eigen.
 *
 * Copyright (C) 2026 SFWA contributors
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/* Link with -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free to
 * fail immediately if any heap allocation is invoked on bare-metal Cortex-M. */

#include <stddef.h>

extern void print(const char* s);
extern void finish_code(int code);

static void heap_trap(const char* fn) {
  print("HEAP_TRAP: ");
  print(fn);
  print("\n");
  finish_code(3);
}

void* __wrap_malloc(size_t n) {
  (void)n;
  heap_trap("malloc");
  return 0;
}

void* __wrap_calloc(size_t nmemb, size_t size) {
  (void)nmemb;
  (void)size;
  heap_trap("calloc");
  return 0;
}

void* __wrap_realloc(void* ptr, size_t size) {
  (void)ptr;
  (void)size;
  heap_trap("realloc");
  return 0;
}

void __wrap_free(void* ptr) {
  if (ptr != 0) {
    heap_trap("free");
  }
}
