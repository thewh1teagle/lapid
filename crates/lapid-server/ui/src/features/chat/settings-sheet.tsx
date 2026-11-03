import { zodResolver } from "@hookform/resolvers/zod";
import { RotateCcwIcon, SlidersHorizontalIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { Controller, useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Input, Textarea } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Switch } from "@/components/ui/switch";
import { DEFAULT_SAMPLING, type Sampling, useSettings } from "@/stores/settings";
import { useConversations } from "@/stores/conversations";
import { SteeringSection } from "./steering-section";

const schema = z.object({
  temperature: z.number({ error: "Required" }).min(0).max(2),
  top_p: z.number({ error: "Required" }).gt(0).max(1),
  top_k: z.number({ error: "Required" }).int().min(0).max(1000),
  max_tokens: z.number({ error: "Required" }).int().min(1).max(262144),
  repetition_penalty: z.number({ error: "Required" }).min(0.5).max(2),
  seed: z.string().regex(/^\d*$/, "Whole number or empty"),
  stop: z.string().max(500),
  enable_thinking: z.boolean(),
}) satisfies z.ZodType<Sampling>;

type NumKey = "temperature" | "top_p" | "top_k" | "max_tokens" | "repetition_penalty";
const NUMERIC: { key: NumKey; label: string; step: number; hint: string }[] = [
  { key: "temperature", label: "Temperature", step: 0.05, hint: "0 to 2. Higher is more random, 0 is greedy." },
  { key: "top_p", label: "Top P", step: 0.01, hint: "Nucleus sampling mass, 0 to 1." },
  { key: "top_k", label: "Top K", step: 1, hint: "0 disables." },
  { key: "max_tokens", label: "Max tokens", step: 1, hint: "Upper bound on generated tokens." },
  { key: "repetition_penalty", label: "Repetition penalty", step: 0.01, hint: "1 disables." },
];

export function SettingsSheet({ convId }: { convId: string | null }) {
  const [open, setOpen] = useState(false);
  const sampling = useSettings((s) => s.sampling);
  const setSampling = useSettings((s) => s.setSampling);
  const conv = useConversations((s) => s.conversations.find((c) => c.id === convId));
  const setSystemPrompt = useConversations((s) => s.setSystemPrompt);
  const draftSystem = useConversations((s) => s.draftSystem);
  const setDraftSystem = useConversations((s) => s.setDraftSystem);

  const form = useForm<Sampling>({ resolver: zodResolver(schema), defaultValues: sampling, mode: "onChange" });
  // Reset only when the sheet opens; resetting on every store update would fight the user's typing.
  useEffect(() => {
    if (open) form.reset(useSettings.getState().sampling);
  }, [open, form]);

  // Persist every valid change immediately; no separate save step.
  useEffect(() => {
    const sub = form.watch(() => {
      void form.handleSubmit((v) => setSampling(v))();
    });
    return () => sub.unsubscribe();
  }, [form, setSampling]);

  const errors = form.formState.errors;
  const system = conv ? conv.systemPrompt : draftSystem;

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="outline" size="icon" aria-label="Chat settings" title="Chat settings">
          <SlidersHorizontalIcon />
        </Button>
      </SheetTrigger>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>Chat settings</SheetTitle>
          <SheetDescription>System prompt applies to this conversation. Sampling applies to every chat.</SheetDescription>
        </SheetHeader>
        <form className="flex-1 space-y-5 overflow-y-auto px-5 pb-6" onSubmit={(e) => e.preventDefault()}>
          <div className="space-y-2">
            <Label htmlFor="system-prompt">System prompt</Label>
            <Textarea
              id="system-prompt"
              value={system}
              placeholder="You are a helpful assistant."
              className="min-h-24"
              onChange={(e) => (conv ? setSystemPrompt(conv.id, e.target.value) : setDraftSystem(e.target.value))}
            />
          </div>
          <div className="flex items-center justify-between gap-4 rounded-lg border p-3">
            <div className="space-y-1">
              <Label htmlFor="enable-thinking">Enable thinking</Label>
              <p className="text-xs text-muted-foreground">Lets reasoning models think before answering.</p>
            </div>
            <Controller
              control={form.control}
              name="enable_thinking"
              render={({ field }) => <Switch id="enable-thinking" checked={field.value} onCheckedChange={field.onChange} />}
            />
          </div>
          <SteeringSection />
          <div className="grid grid-cols-2 gap-4">
            {NUMERIC.map(({ key, label, step, hint }) => (
              <div key={key} className="space-y-1.5">
                <Label htmlFor={key}>{label}</Label>
                <Input id={key} type="number" step={step} inputMode="decimal" aria-invalid={!!errors[key]} {...form.register(key, { valueAsNumber: true })} />
                <p className={errors[key] ? "text-xs text-destructive" : "text-xs text-muted-foreground"}>{errors[key]?.message ?? hint}</p>
              </div>
            ))}
            <div className="space-y-1.5">
              <Label htmlFor="seed">Seed</Label>
              <Input id="seed" inputMode="numeric" placeholder="random" aria-invalid={!!errors.seed} {...form.register("seed")} />
              <p className={errors.seed ? "text-xs text-destructive" : "text-xs text-muted-foreground"}>{errors.seed?.message ?? "Empty for random."}</p>
            </div>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="stop">Stop sequences</Label>
            <Textarea id="stop" placeholder="One per line" className="min-h-16 font-mono text-xs" {...form.register("stop")} />
            <p className="text-xs text-muted-foreground">One sequence per line. Write \n for a newline.</p>
          </div>
          <Button type="button" variant="outline" size="sm" onClick={() => form.reset(DEFAULT_SAMPLING)}>
            <RotateCcwIcon /> Reset to defaults
          </Button>
        </form>
      </SheetContent>
    </Sheet>
  );
}
