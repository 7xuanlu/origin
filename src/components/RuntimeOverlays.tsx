import { MilestoneToaster } from "./onboarding/MilestoneToaster";
import UpdaterDialog from "./UpdaterDialog";

type RuntimeOverlaysProps = {
  readonly review?: boolean;
};

export function RuntimeOverlays({ review = __WENLAN_REVIEW__ }: RuntimeOverlaysProps) {
  if (review) return null;

  return (
    <>
      <MilestoneToaster />
      <UpdaterDialog />
    </>
  );
}
