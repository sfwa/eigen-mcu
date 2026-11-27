/*
 * This file is part of eigen-mcu, an MCU-targeted fork of Eigen.
 *
 * Copyright (C) 2026 SFWA contributors
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/* Bare-metal Cortex-M startup for QEMU mps2-an500 (Cortex-M7) and mps2-an386
 * (Cortex-M4F). Provides vector table, FPU enable, ARM semihosting I/O and
 * exit-code propagation, SysTick instruction counting under -icount shift=0
 * (25 MHz SysTick = 40 instructions per tick), and stack high-water marking. */

#include <stdint.h>

extern uint32_t _estack, _sstack, _sidata, _sdata, _edata, _sbss, _ebss;
int main(void);
void __libc_init_array(void);

#define SYST_CSR (*(volatile uint32_t*)0xE000E010u)
#define SYST_RVR (*(volatile uint32_t*)0xE000E014u)
#define SYST_CVR (*(volatile uint32_t*)0xE000E018u)
#define CPACR (*(volatile uint32_t*)0xE000ED88u)

static volatile uint32_t g_systick_wraps = 0;
static uint32_t* g_paint_top = 0;

static int semihost(int op, void* arg) {
  register int r0 __asm__("r0") = op;
  register void* r1 __asm__("r1") = arg;
  __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
  return r0;
}

void print(const char* s) { semihost(0x04, (void*)s); }

void print_u(uint64_t v) {
  char buf[24];
  int i = 23;
  buf[i] = '\0';
  do {
    buf[--i] = (char)('0' + (v % 10u));
    v /= 10u;
  } while (v != 0u);
  print(&buf[i]);
}

void finish_code(int code) {
  /* ARM semihosting SYS_EXIT_EXTENDED (0x20) propagates exit status to QEMU. */
  uint32_t block[2];
  block[0] = 0x20026u; /* ADP_Stopped_ApplicationExit */
  block[1] = (uint32_t)code;
  semihost(0x20, block);
  /* Fallback to SYS_EXIT (0x18) if extended exit is unsupported. */
  semihost(0x18, (void*)(uintptr_t)0x20026u);
  for (;;) {
  }
}

void finish(void) { finish_code(0); }

void SysTick_Handler(void) { g_systick_wraps++; }

void Fault_Handler(void) {
  print("FAULT\n");
  finish_code(2);
}

void timer_start(void) {
  SYST_CSR = 0u;
  g_systick_wraps = 0u;
  SYST_RVR = 0x00FFFFFFu;
  SYST_CVR = 0u;
  SYST_CSR = 7u; /* ENABLE | TICKINT | CLKSOURCE */
  /* Wait for the initial reload and first tick before measuring. */
  while (SYST_CVR == 0u) {
  }
  uint32_t first = SYST_CVR;
  while (SYST_CVR == first) {
  }
}

/* Under QEMU -icount shift=0, virtual time advances 1 ns per instruction.
 * Both mps2-an500 and mps2-an386 clock SysTick at 25 MHz (40 ns/tick). */
uint64_t instructions(void) {
  uint32_t before, after, value;
  do {
    before = g_systick_wraps;
    value = SYST_CVR;
    after = g_systick_wraps;
  } while (before != after);
  return ((((uint64_t)before) << 24) + (0x00FFFFFFu - value)) * 40u;
}

__attribute__((noinline)) void stack_paint(void) {
  uint32_t* top = (uint32_t*)__builtin_frame_address(0) - 32;
  for (uint32_t* p = &_sstack; p < top; ++p) {
    *p = 0xA5A5A5A5u;
  }
  g_paint_top = top;
}

uint32_t stack_used(void) {
  uint32_t* p = &_sstack;
  while (p < g_paint_top && *p == 0xA5A5A5A5u) {
    ++p;
  }
  return (uint32_t)((uint8_t*)&_estack - (uint8_t*)p);
}

void _init(void) {}
void _fini(void) {}

void Reset_Handler(void) {
#if defined(__ARM_FP) && (__ARM_FP > 0)
  CPACR |= (0xFu << 20); /* Enable CP10 and CP11 full access for FPU. */
  __asm__ volatile("dsb\n isb");
#endif
  uint32_t* src = &_sidata;
  for (uint32_t* dst = &_sdata; dst < &_edata; ++dst) {
    *dst = *src++;
  }
  for (uint32_t* dst = &_sbss; dst < &_ebss; ++dst) {
    *dst = 0u;
  }
  __libc_init_array();
  int rc = main();
  finish_code(rc);
}

__attribute__((section(".vectors"), used)) void (*const vectors[16])(void) = {
    (void (*)(void))&_estack,
    Reset_Handler,
    Fault_Handler,
    Fault_Handler,
    Fault_Handler,
    Fault_Handler,
    Fault_Handler,
    0,
    0,
    0,
    0,
    Fault_Handler,
    Fault_Handler,
    0,
    Fault_Handler,
    SysTick_Handler,
};
