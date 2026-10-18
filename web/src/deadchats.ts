/**
 * Pure set of chat ids deleted on disk. A finishing stream must not write
 * a deleted chat back, and a failed delete must unmark the id so the chat
 * keeps saving. No DOM and no storage.
 */

/** Dead-id marks with mark/unmark/has so delete failure is testable. */
export class DeadChats {
  private readonly ids = new Set<string>();

  /** Mark a chat id as deleted. */
  mark(id: string): void {
    this.ids.add(id);
  }

  /** Undo a mark: the delete failed and the chat still exists. */
  unmark(id: string): void {
    this.ids.delete(id);
  }

  /** True when the chat id was deleted. */
  has(id: string): boolean {
    return this.ids.has(id);
  }
}
