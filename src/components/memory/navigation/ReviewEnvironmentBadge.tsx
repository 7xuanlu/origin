import { useTranslation } from "react-i18next";

import "./review-environment-badge.css";

type ReviewEnvironmentBadgeProps = {
  readonly compact?: boolean;
  readonly enabled?: boolean;
  readonly onReset?: () => void;
};

function resetReviewEnvironment() {
  window.localStorage.clear();
  window.location.reload();
}

export function ReviewEnvironmentBadge({
  compact = false,
  enabled = __WENLAN_REVIEW__,
  onReset = resetReviewEnvironment,
}: ReviewEnvironmentBadgeProps) {
  const { t } = useTranslation();

  if (!enabled) return null;

  return (
    <div
      aria-label={t("reviewEnvironment.fixtureNotice")}
      className={`review-environment${compact ? " review-environment-compact" : ""}`}
      data-review-environment="fixture-only"
      role="status"
    >
      <div className="review-environment-heading">
        <span className="review-environment-stamp">
          {t("reviewEnvironment.testData")}
        </span>
        {!compact && (
          <button
            aria-label={t("reviewEnvironment.reset")}
            className="review-environment-reset"
            onClick={onReset}
            title={t("reviewEnvironment.reset")}
            type="button"
          >
            <svg aria-hidden="true" fill="none" height="12" viewBox="0 0 24 24" width="12">
              <path d="M20 7v5h-5M4 17v-5h5M6.1 9a7 7 0 0111.5-2.4L20 9M4 15l2.4 2.4A7 7 0 0017.9 15" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" />
            </svg>
          </button>
        )}
      </div>
      {!compact && (
        <span className="review-environment-notice">
          {t("reviewEnvironment.fixtureNotice")}
        </span>
      )}
    </div>
  );
}
