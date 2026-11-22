export interface ParsedSseEvent {
  event: string;
  data: string;
}

/**
 * Parses a buffer of Server-Sent Events (SSE) text frames into `{ event, data }` objects
 * and returns the unparsed trailing fragment for incremental stream decoding.
 */
export function parseSseChunks(buffer: string): {
  events: ParsedSseEvent[];
  remainder: string;
} {
  const normalized = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const blocks = normalized.split('\n\n');
  const remainder = blocks.pop() ?? '';
  const events: ParsedSseEvent[] = [];

  for (const block of blocks) {
    if (!block.trim()) continue;
    let eventName = 'message';
    const dataLines: string[] = [];
    for (const line of block.split('\n')) {
      if (line.startsWith(':')) continue;
      if (line.startsWith('event:')) {
        eventName = line.slice('event:'.length).trim() || 'message';
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice('data:'.length).trimStart());
      }
    }
    if (dataLines.length > 0) {
      events.push({ event: eventName, data: dataLines.join('\n') });
    }
  }

  return { events, remainder };
}
