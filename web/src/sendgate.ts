/**
 * Pure in-flight guard for the composer. send() closes the gate
 * synchronously, before its first await, so a second send, a chat switch,
 * or a delete cannot start a second stream. No DOM and no storage.
 */

/** One send at a time; closed at the top of send(), reopened when it ends. */
export class SendGate {
  private busy = false;

  /** Close the gate; false when a send is already in flight. */
  tryBegin(): boolean {
    if (this.busy) return false;
    this.busy = true;
    return true;
  }

  /** Reopen the gate after the send ended: finish, abort, or failure. */
  end(): void {
    this.busy = false;
  }
}
