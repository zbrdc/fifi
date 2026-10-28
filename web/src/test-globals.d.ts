/**
 * Minimal ambient declarations for the Node built-in test modules the unit
 * tests import. The devDependencies stay exactly typescript and vite.
 */

declare module "node:test" {
  export function test(name: string, fn: () => void | Promise<void>): void;
}

declare module "node:assert" {
  export function strictEqual(actual: unknown, expected: unknown, message?: string): void;
  export function deepStrictEqual(actual: unknown, expected: unknown, message?: string): void;
  export function ok(value: unknown, message?: string): void;
}
