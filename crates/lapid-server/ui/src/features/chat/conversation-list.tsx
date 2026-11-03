import { EllipsisIcon, EyeOffIcon, PencilIcon, PlusIcon, Trash2Icon } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { useConversations, useStream } from "@/stores/conversations";

export function ConversationList({ onSelect }: { onSelect?: () => void }) {
  const conversations = useConversations((s) => s.conversations);
  const activeId = useConversations((s) => s.activeId);
  const select = useConversations((s) => s.select);
  const rename = useConversations((s) => s.rename);
  const remove = useConversations((s) => s.remove);
  const streamingConv = useStream((s) => s.convId);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);

  const commit = () => {
    if (editing) rename(editing, draft);
    setEditing(null);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col border-t">
      <div className="flex items-center justify-between px-4 pt-3 pb-1">
        <span className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Conversations</span>
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label="New chat"
          title="New chat"
          onClick={() => {
            select(null);
            onSelect?.();
          }}
        >
          <PlusIcon />
        </Button>
      </div>
      <ul className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {sorted.length === 0 && <li className="px-3 py-2 text-xs text-muted-foreground">No conversations yet.</li>}
        {sorted.map((c) => (
          <li key={c.id} className="group relative">
            {editing === c.id ? (
              <form
                className="px-1 py-0.5"
                onSubmit={(e) => {
                  e.preventDefault();
                  commit();
                }}
              >
                <Input
                  autoFocus
                  value={draft}
                  aria-label="Conversation title"
                  className="h-8"
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={commit}
                  onKeyDown={(e) => e.key === "Escape" && setEditing(null)}
                />
              </form>
            ) : (
              <>
                <button
                  type="button"
                  onClick={() => {
                    select(c.id);
                    onSelect?.();
                  }}
                  onDoubleClick={() => {
                    setEditing(c.id);
                    setDraft(c.title);
                  }}
                  className={cn(
                    "flex h-8 w-full items-center gap-2 rounded-md pr-8 pl-3 text-left text-sm outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
                    c.id === activeId ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                  )}
                >
                  {streamingConv === c.id && <span className="size-1.5 shrink-0 animate-pulse rounded-full bg-brand" />}
                  {c.ephemeral && <EyeOffIcon aria-label="Private chat (not saved)" className="size-3.5 shrink-0" />}
                  <span className="truncate">{c.title}</span>
                </button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      aria-label={`Actions for ${c.title}`}
                      className="absolute top-0.5 right-0.5 size-7 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100 max-md:opacity-100"
                    >
                      <EllipsisIcon />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start" side="right">
                    <DropdownMenuItem
                      onSelect={() => {
                        setEditing(c.id);
                        setDraft(c.title);
                      }}
                    >
                      <PencilIcon /> Rename
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      variant="destructive"
                      onSelect={() => {
                        if (streamingConv === c.id) useStream.getState().stop();
                        remove(c.id);
                      }}
                    >
                      <Trash2Icon /> Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
