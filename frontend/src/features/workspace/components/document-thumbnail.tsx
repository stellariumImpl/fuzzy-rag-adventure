import { useEffect, useState } from "react";
import { FileText } from "lucide-react";
import { getDocumentThumbnailUrl } from "../utils";

interface DocumentThumbnailProps {
  docId: string;
  docName: string;
  width: number;
  className: string;
  iconSize: number;
}

export function DocumentThumbnail({
  docId,
  docName,
  width,
  className,
  iconSize,
}: DocumentThumbnailProps) {
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    setHasError(false);
  }, [docId]);

  if (hasError) {
    return (
      <div
        className={`${className} flex items-center justify-center bg-[#d7dee7] text-[#93a0af]`}
      >
        <FileText size={iconSize} />
      </div>
    );
  }

  return (
    <img
      src={getDocumentThumbnailUrl(docId, width)}
      alt={`${docName} cover`}
      className={`${className} bg-[#d7dee7] object-cover`}
      loading="lazy"
      onError={() => setHasError(true)}
    />
  );
}
