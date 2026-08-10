import { MilestoneToaster } from "./onboarding/MilestoneToaster";
import UpdaterDialog from "./UpdaterDialog";

type RuntimeOverlaysProps = {
  readonly review?: boolean;
  readonly variant?: "main" | "updater-only";
};

export function RuntimeOverlays({
  review = __WENLAN_REVIEW__,
  variant = "main",
}: RuntimeOverlaysProps) {
  if (review) return null;

  return (
    <>
      {variant === "main" && <MilestoneToaster />}
      <UpdaterDialog />
    </>
  );
}
