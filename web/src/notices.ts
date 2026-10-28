/**
 * Pure notice queue policy: one banner per distinct message text, the
 * newest entry last, and the queue capped so banners cannot stack over
 * the composer. No DOM; main.ts renders the list.
 */

const NOTICE_CAP = 3;

/** The queue after showing `text`: its earlier copy is dropped, the text
 *  goes last, and the oldest entry beyond the cap falls off. */
export function noticeList(current: readonly string[], text: string): string[] {
  const next = current.filter((t) => t !== text);
  next.push(text);
  return next.length > NOTICE_CAP ? next.slice(next.length - NOTICE_CAP) : next;
}
