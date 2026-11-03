import { useModels } from "@/hooks/use-server";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import type { SteeringInfo } from "@/lib/api";
import { type SteerChoice, useSettings } from "@/stores/settings";

function defaults(files: SteeringInfo[]): Record<string, SteerChoice> {
  return Object.fromEntries(files.map((f) => [f.name, { enabled: f.default_scale !== 0, scale: f.default_scale !== 0 ? f.default_scale : 1 }]));
}

/** Activation steering files of the selected chat model: on/off and multiplier each. */
export function SteeringSection() {
  const { models } = useModels("chat");
  const modelId = useSettings((s) => s.models.chat);
  const model = models.find((m) => m.id === modelId) ?? models[0];
  const stored = useSettings((s) => (model ? s.steer[model.id] : undefined));
  const setSteer = useSettings((s) => s.setSteer);
  const files = model?.steering ?? [];
  if (!model || files.length === 0) return null;
  const choices = { ...defaults(files), ...stored };
  const update = (name: string, patch: Partial<SteerChoice>) => setSteer(model.id, { ...choices, [name]: { ...choices[name], ...patch } });

  return (
    <div className="space-y-3 rounded-lg border p-3">
      <div className="flex items-center justify-between gap-2">
        <Label>Steering</Label>
        {stored && (
          <button type="button" className="text-xs text-muted-foreground underline" onClick={() => setSteer(model.id, undefined)}>
            Server default
          </button>
        )}
      </div>
      {files.map((f) => {
        const c = choices[f.name];
        const id = `steer-${f.name}`;
        return (
          <div key={f.name} className="space-y-1.5">
            <div className="flex items-center justify-between gap-3">
              <Label htmlFor={id} className="truncate font-mono text-xs" title={f.name}>
                {f.name}
              </Label>
              <div className="flex items-center gap-2">
                <Input
                  aria-label={`${f.name} scale`}
                  type="number"
                  step={0.1}
                  inputMode="decimal"
                  className="h-7 w-20"
                  value={c.scale}
                  disabled={!c.enabled}
                  onChange={(e) => {
                    const v = e.target.valueAsNumber;
                    if (Number.isFinite(v)) update(f.name, { scale: v });
                  }}
                />
                <Switch id={id} checked={c.enabled} onCheckedChange={(enabled) => update(f.name, { enabled })} />
              </div>
            </div>
            {f.description && <p className="text-xs text-muted-foreground">{f.description}</p>}
          </div>
        );
      })}
      <p className="text-xs text-muted-foreground">Multiplies each file's own strength (1 = as made). Changing it recomputes the conversation's cached prompt.</p>
    </div>
  );
}
