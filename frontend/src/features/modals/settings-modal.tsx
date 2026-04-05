import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Building2,
  FlaskConical,
  Link2,
  Save,
  Settings as SettingsIcon,
  Shield,
  X,
} from "lucide-react";
import { Input } from "../../components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../components/ui/select";
import { Checkbox } from "../../components/ui/checkbox";
import {
  getAssistantSettings,
  getAssistantRetrievalStatus,
  getLlamaParseSettings,
  getMultimodalOCRSettings,
  listEmbeddingProviders,
  listAssistantProviders,
  listMultimodalOCRProviders,
  testAssistantConnection,
  updateLlamaParseSettings,
  updateMultimodalOCRSettings,
  updateAssistantSettings,
  type LlamaParseSettings,
  type MultimodalOCRSettings,
  type AssistantProviderOption,
  type AssistantSettings,
  type RetrievalPipelineStatus,
} from "../../services/workspace-api";

interface SettingsModalProps {
  open: boolean;
  isMobileViewport: boolean;
  theme: "light" | "dark";
  onClose: () => void;
}

type SettingsTab = "assistant" | "pdf-table-parse";

type AssistantFormState = {
  llmProvider: string;
  llmModel: string;
  apiBase: string;
  apiKeyInput: string;
  apiKeyConfigured: boolean;
  apiKeyPreview: string;
  temperature: string;
  maxTokens: string;
  embeddingProvider: string;
  embeddingModel: string;
  embeddingApiBase: string;
  embeddingApiKeyInput: string;
  embeddingApiKeyConfigured: boolean;
  embeddingApiKeyPreview: string;
  updatedAt: string;
};

type LlamaParseFormState = {
  enabled: boolean;
  baseUrl: string;
  targetPages: string;
  resultType: string;
  splitByPage: boolean;
  apiKeyInput: string;
  apiKeyConfigured: boolean;
  apiKeyPreview: string;
  updatedAt: string;
};

type MultimodalOCRFormState = {
  enabled: boolean;
  provider: string;
  model: string;
  apiBase: string;
  apiKeyInput: string;
  apiKeyConfigured: boolean;
  apiKeyPreview: string;
  maxPagesPerDoc: string;
  strategy: "fallback" | "prefer_ocr" | "hybrid";
  prompt: string;
  updatedAt: string;
};

const DEFAULT_FORM: AssistantFormState = {
  llmProvider: "dashscope",
  llmModel: "qwen-max",
  apiBase: "https://api.dashscope.aliyuncs.com",
  apiKeyInput: "",
  apiKeyConfigured: false,
  apiKeyPreview: "",
  temperature: "0.2",
  maxTokens: "512",
  embeddingProvider: "dashscope",
  embeddingModel: "text-embedding-v4",
  embeddingApiBase: "https://api.dashscope.aliyuncs.com",
  embeddingApiKeyInput: "",
  embeddingApiKeyConfigured: false,
  embeddingApiKeyPreview: "",
  updatedAt: "",
};

const DEFAULT_LLAMAPARSE_FORM: LlamaParseFormState = {
  enabled: false,
  baseUrl: "https://api.cloud.llamaindex.ai",
  targetPages: "",
  resultType: "markdown",
  splitByPage: true,
  apiKeyInput: "",
  apiKeyConfigured: false,
  apiKeyPreview: "",
  updatedAt: "",
};

const DEFAULT_MULTIMODAL_OCR_FORM: MultimodalOCRFormState = {
  enabled: false,
  provider: "openai",
  model: "gpt-4o-mini",
  apiBase: "https://api.openai.com/v1",
  apiKeyInput: "",
  apiKeyConfigured: false,
  apiKeyPreview: "",
  maxPagesPerDoc: "8",
  strategy: "hybrid",
  prompt: "",
  updatedAt: "",
};

const TEMPERATURE_OPTIONS = [
  "0.0",
  "0.1",
  "0.2",
  "0.3",
  "0.5",
  "0.7",
  "1.0",
  "1.2",
  "1.5",
  "2.0",
];

function mapAssistantSettingsToForm(
  settings: AssistantSettings,
  llmProviderOptions: AssistantProviderOption[],
  embeddingProviderOptions: AssistantProviderOption[],
): AssistantFormState {
  const firstLlmProvider = llmProviderOptions[0];
  const currentLlmProvider =
    llmProviderOptions.find((option) => option.provider === settings.llm_provider) ??
    firstLlmProvider;
  const firstEmbeddingProvider = embeddingProviderOptions[0];
  const currentEmbeddingProvider =
    embeddingProviderOptions.find(
      (option) => option.provider === settings.embedding_provider,
    ) ?? firstEmbeddingProvider;

  return {
    llmProvider: currentLlmProvider?.provider || settings.llm_provider || "dashscope",
    llmModel: settings.llm_model || currentLlmProvider?.default_model || "qwen-max",
    apiBase:
      settings.api_base ||
      currentLlmProvider?.default_api_base ||
      "https://api.dashscope.aliyuncs.com",
    apiKeyInput: "",
    apiKeyConfigured: settings.api_key_configured,
    apiKeyPreview: settings.api_key_preview || "",
    temperature: String(settings.temperature ?? 0.2),
    maxTokens: String(settings.max_tokens ?? 512),
    embeddingProvider:
      currentEmbeddingProvider?.provider || settings.embedding_provider || "dashscope",
    embeddingModel:
      settings.embedding_model ||
      currentEmbeddingProvider?.default_model ||
      "text-embedding-v4",
    embeddingApiBase:
      settings.embedding_api_base ||
      currentEmbeddingProvider?.default_api_base ||
      "https://api.dashscope.aliyuncs.com",
    embeddingApiKeyInput: "",
    embeddingApiKeyConfigured: settings.embedding_api_key_configured,
    embeddingApiKeyPreview: settings.embedding_api_key_preview || "",
    updatedAt: settings.updated_at || "",
  };
}

function mapLlamaParseSettingsToForm(
  settings: LlamaParseSettings,
): LlamaParseFormState {
  return {
    enabled: settings.enabled,
    baseUrl: settings.base_url || "https://api.cloud.llamaindex.ai",
    targetPages: settings.target_pages || "",
    resultType: settings.result_type || "markdown",
    splitByPage: settings.split_by_page,
    apiKeyInput: "",
    apiKeyConfigured: settings.api_key_configured,
    apiKeyPreview: settings.api_key_preview || "",
    updatedAt: settings.updated_at || "",
  };
}

function mapMultimodalOCRSettingsToForm(
  settings: MultimodalOCRSettings,
  providerOptions: AssistantProviderOption[],
): MultimodalOCRFormState {
  const firstProvider = providerOptions[0];
  const currentProvider =
    providerOptions.find((option) => option.provider === settings.provider) ??
    firstProvider;

  const rawStrategy = (settings.strategy || "hybrid").toLowerCase();
  const strategy: "fallback" | "prefer_ocr" | "hybrid" =
    rawStrategy === "fallback" || rawStrategy === "prefer_ocr" || rawStrategy === "hybrid"
      ? rawStrategy
      : "hybrid";

  return {
    enabled: settings.enabled,
    provider: currentProvider?.provider || settings.provider || "openai",
    model: settings.model || currentProvider?.default_model || "gpt-4o-mini",
    apiBase:
      settings.api_base ||
      currentProvider?.default_api_base ||
      "https://api.openai.com/v1",
    apiKeyInput: "",
    apiKeyConfigured: settings.api_key_configured,
    apiKeyPreview: settings.api_key_preview || "",
    maxPagesPerDoc: String(settings.max_pages_per_doc ?? 8),
    strategy,
    prompt: settings.prompt || "",
    updatedAt: settings.updated_at || "",
  };
}

export function SettingsModal({
  open,
  isMobileViewport,
  theme,
  onClose,
}: SettingsModalProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>("assistant");
  const [assistantForm, setAssistantForm] = useState<AssistantFormState>(DEFAULT_FORM);
  const [llamaParseForm, setLlamaParseForm] = useState<LlamaParseFormState>(
    DEFAULT_LLAMAPARSE_FORM,
  );
  const [multimodalOCRForm, setMultimodalOCRForm] =
    useState<MultimodalOCRFormState>(DEFAULT_MULTIMODAL_OCR_FORM);
  const [providerOptions, setProviderOptions] = useState<AssistantProviderOption[]>([]);
  const [embeddingProviderOptions, setEmbeddingProviderOptions] = useState<
    AssistantProviderOption[]
  >([]);
  const [ocrProviderOptions, setOcrProviderOptions] = useState<
    AssistantProviderOption[]
  >([]);
  const [retrievalStatus, setRetrievalStatus] = useState<RetrievalPipelineStatus | null>(
    null,
  );
  const [loadingAssistant, setLoadingAssistant] = useState(false);
  const [loadingLlamaParse, setLoadingLlamaParse] = useState(false);
  const [loadingMultimodalOCR, setLoadingMultimodalOCR] = useState(false);
  const [switchingProvider, setSwitchingProvider] = useState(false);
  const [savingAssistant, setSavingAssistant] = useState(false);
  const [savingLlamaParse, setSavingLlamaParse] = useState(false);
  const [savingMultimodalOCR, setSavingMultimodalOCR] = useState(false);
  const [testingAssistant, setTestingAssistant] = useState(false);
  const [assistantError, setAssistantError] = useState<string | null>(null);
  const [assistantNotice, setAssistantNotice] = useState<string | null>(null);
  const [llamaParseError, setLlamaParseError] = useState<string | null>(null);
  const [llamaParseNotice, setLlamaParseNotice] = useState<string | null>(null);
  const [multimodalOCRError, setMultimodalOCRError] = useState<string | null>(null);
  const [multimodalOCRNotice, setMultimodalOCRNotice] = useState<string | null>(null);
  const [clearApiKeyOnSave, setClearApiKeyOnSave] = useState(false);
  const [clearEmbeddingApiKeyOnSave, setClearEmbeddingApiKeyOnSave] = useState(false);
  const [clearLlamaParseKeyOnSave, setClearLlamaParseKeyOnSave] = useState(false);
  const [clearMultimodalOCRKeyOnSave, setClearMultimodalOCRKeyOnSave] =
    useState(false);

  const isBusy =
    loadingAssistant ||
    loadingLlamaParse ||
    loadingMultimodalOCR ||
    switchingProvider ||
    savingAssistant ||
    savingLlamaParse ||
    savingMultimodalOCR ||
    testingAssistant;

  const loadAssistantSettings = useCallback(async () => {
    setLoadingAssistant(true);
    setLoadingLlamaParse(true);
    setLoadingMultimodalOCR(true);
    setAssistantError(null);
    setAssistantNotice(null);
    setLlamaParseError(null);
    setLlamaParseNotice(null);
    setMultimodalOCRError(null);
    setMultimodalOCRNotice(null);
    try {
      const [
        providers,
        embeddingProviders,
        ocrProviders,
        settings,
        llamaParseSettings,
        multimodalOCRSettings,
        status,
      ] =
        await Promise.all([
        listAssistantProviders(),
        listEmbeddingProviders(),
        listMultimodalOCRProviders(),
        getAssistantSettings(),
        getLlamaParseSettings(),
        getMultimodalOCRSettings(),
        getAssistantRetrievalStatus(),
      ]);
      const items = providers.items ?? [];
      if (items.length === 0) {
        throw new Error("No LLM providers configured on backend.");
      }
      const embeddingItems = embeddingProviders.items ?? [];
      if (embeddingItems.length === 0) {
        throw new Error("No embedding providers configured on backend.");
      }
      const ocrItems = ocrProviders.items ?? [];
      if (ocrItems.length === 0) {
        throw new Error("No multimodal OCR providers configured on backend.");
      }
      setProviderOptions(items);
      setEmbeddingProviderOptions(embeddingItems);
      setOcrProviderOptions(ocrItems);
      setAssistantForm(mapAssistantSettingsToForm(settings, items, embeddingItems));
      setClearApiKeyOnSave(false);
      setClearEmbeddingApiKeyOnSave(false);
      setLlamaParseForm(mapLlamaParseSettingsToForm(llamaParseSettings));
      setClearLlamaParseKeyOnSave(false);
      setMultimodalOCRForm(
        mapMultimodalOCRSettingsToForm(multimodalOCRSettings, ocrItems),
      );
      setClearMultimodalOCRKeyOnSave(false);
      setRetrievalStatus(status);
    } catch (error) {
      const reason = error instanceof Error ? error.message : "Unknown error";
      setAssistantError(`Failed to load assistant settings: ${reason}`);
      setLlamaParseError(`Failed to load LlamaParse settings: ${reason}`);
      setMultimodalOCRError(
        `Failed to load multimodal OCR settings: ${reason}`,
      );
    } finally {
      setLoadingAssistant(false);
      setLoadingLlamaParse(false);
      setLoadingMultimodalOCR(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    void loadAssistantSettings();
  }, [open, loadAssistantSettings]);

  const loadProviderSettings = useCallback(
    async (options: { llmProvider?: string; embeddingProvider?: string }) => {
      setSwitchingProvider(true);
      setAssistantError(null);
      setAssistantNotice(null);
      try {
        const settings = await getAssistantSettings(options);
        setAssistantForm(
          mapAssistantSettingsToForm(settings, providerOptions, embeddingProviderOptions),
        );
        setClearApiKeyOnSave(false);
        setClearEmbeddingApiKeyOnSave(false);
      } catch (error) {
        const reason = error instanceof Error ? error.message : "Unknown error";
        setAssistantError(`Failed to load provider settings: ${reason}`);
      } finally {
        setSwitchingProvider(false);
      }
    },
    [embeddingProviderOptions, providerOptions],
  );

  const buildPayload = useCallback(() => {
    const selectedProvider =
      providerOptions.find((option) => option.provider === assistantForm.llmProvider) ??
      providerOptions[0];
    const selectedEmbeddingProvider =
      embeddingProviderOptions.find(
        (option) => option.provider === assistantForm.embeddingProvider,
      ) ?? embeddingProviderOptions[0];

    if (!selectedProvider) {
      throw new Error("No provider is available.");
    }
    if (!selectedEmbeddingProvider) {
      throw new Error("No embedding provider is available.");
    }

    const temperature = Number(assistantForm.temperature);
    if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
      throw new Error("Temperature must be a number between 0 and 2.");
    }

    const maxTokens = Number(assistantForm.maxTokens);
    if (!Number.isFinite(maxTokens) || maxTokens < 64 || maxTokens > 4096) {
      throw new Error("Max tokens must be an integer between 64 and 4096.");
    }

    const payload: {
      llm_provider: string;
      llm_model: string;
      api_base: string;
      temperature: number;
      max_tokens: number;
      api_key?: string;
      embedding_provider: string;
      embedding_model: string;
      embedding_api_base: string;
      embedding_api_key?: string;
    } = {
      llm_provider: assistantForm.llmProvider,
      llm_model: assistantForm.llmModel.trim() || selectedProvider.default_model,
      api_base: assistantForm.apiBase.trim() || selectedProvider.default_api_base,
      temperature,
      max_tokens: Math.round(maxTokens),
      embedding_provider: assistantForm.embeddingProvider,
      embedding_model:
        assistantForm.embeddingModel.trim() || selectedEmbeddingProvider.default_model,
      embedding_api_base:
        assistantForm.embeddingApiBase.trim() ||
        selectedEmbeddingProvider.default_api_base,
    };

    const candidateApiKey = assistantForm.apiKeyInput.trim();
    if (candidateApiKey.length > 0) {
      payload.api_key = candidateApiKey;
    } else if (clearApiKeyOnSave) {
      payload.api_key = "";
    }
    const candidateEmbeddingApiKey = assistantForm.embeddingApiKeyInput.trim();
    if (candidateEmbeddingApiKey.length > 0) {
      payload.embedding_api_key = candidateEmbeddingApiKey;
    } else if (clearEmbeddingApiKeyOnSave) {
      payload.embedding_api_key = "";
    }
    return payload;
  }, [
    assistantForm,
    clearApiKeyOnSave,
    clearEmbeddingApiKeyOnSave,
    embeddingProviderOptions,
    providerOptions,
  ]);

  const saveAssistantSettings = useCallback(async () => {
    setSavingAssistant(true);
    setAssistantError(null);
    setAssistantNotice(null);
    try {
      const updated = await updateAssistantSettings(buildPayload());
      setAssistantForm(
        mapAssistantSettingsToForm(updated, providerOptions, embeddingProviderOptions),
      );
      setClearApiKeyOnSave(false);
      setClearEmbeddingApiKeyOnSave(false);
      try {
        const status = await getAssistantRetrievalStatus();
        setRetrievalStatus(status);
      } catch {
        // keep current status display if status refresh fails
      }
      setAssistantNotice("Assistant settings saved.");
    } catch (error) {
      const reason = error instanceof Error ? error.message : "Unknown error";
      setAssistantError(`Failed to save assistant settings: ${reason}`);
    } finally {
      setSavingAssistant(false);
    }
  }, [buildPayload, embeddingProviderOptions, providerOptions]);

  const runAssistantConnectionTest = useCallback(async () => {
    setTestingAssistant(true);
    setAssistantError(null);
    setAssistantNotice(null);
    try {
      const result = await testAssistantConnection(buildPayload());
      const latency = result.latency_ms == null ? "" : ` (${Math.max(0, result.latency_ms)} ms)`;
      if (result.ok) {
        setAssistantNotice(`Connection OK${latency}: ${result.message}`);
      } else {
        setAssistantError(`Connection failed${latency}: ${result.message}`);
      }
    } catch (error) {
      const reason = error instanceof Error ? error.message : "Unknown error";
      setAssistantError(`Connection test request failed: ${reason}`);
    } finally {
      setTestingAssistant(false);
    }
  }, [buildPayload]);

  const buildLlamaParsePayload = useCallback(() => {
    const payload: {
      enabled: boolean;
      base_url: string;
      target_pages: string;
      result_type: string;
      split_by_page: boolean;
      api_key?: string;
    } = {
      enabled: llamaParseForm.enabled,
      base_url: llamaParseForm.baseUrl.trim() || "https://api.cloud.llamaindex.ai",
      target_pages: llamaParseForm.targetPages.trim(),
      result_type: "markdown",
      split_by_page: llamaParseForm.splitByPage,
    };

    const candidateApiKey = llamaParseForm.apiKeyInput.trim();
    if (candidateApiKey.length > 0) {
      payload.api_key = candidateApiKey;
    } else if (clearLlamaParseKeyOnSave) {
      payload.api_key = "";
    }

    return payload;
  }, [clearLlamaParseKeyOnSave, llamaParseForm]);

  const saveLlamaParseSettings = useCallback(async () => {
    setSavingLlamaParse(true);
    setLlamaParseError(null);
    setLlamaParseNotice(null);
    try {
      const updated = await updateLlamaParseSettings(buildLlamaParsePayload());
      setLlamaParseForm(mapLlamaParseSettingsToForm(updated));
      setClearLlamaParseKeyOnSave(false);
      setLlamaParseNotice("LlamaParse settings saved.");
    } catch (error) {
      const reason = error instanceof Error ? error.message : "Unknown error";
      setLlamaParseError(`Failed to save LlamaParse settings: ${reason}`);
    } finally {
      setSavingLlamaParse(false);
    }
  }, [buildLlamaParsePayload]);

  const buildMultimodalOCRPayload = useCallback(() => {
    const selectedOCRProvider =
      ocrProviderOptions.find(
        (option) => option.provider === multimodalOCRForm.provider,
      ) ?? ocrProviderOptions[0];

    if (!selectedOCRProvider) {
      throw new Error("No multimodal OCR provider is available.");
    }

    const maxPagesPerDoc = Number(multimodalOCRForm.maxPagesPerDoc);
    if (!Number.isFinite(maxPagesPerDoc) || maxPagesPerDoc < 1 || maxPagesPerDoc > 128) {
      throw new Error("Max pages per document must be between 1 and 128.");
    }

    const payload: {
      enabled: boolean;
      provider: string;
      model: string;
      api_base: string;
      max_pages_per_doc: number;
      strategy: "fallback" | "prefer_ocr" | "hybrid";
      prompt: string;
      api_key?: string;
    } = {
      enabled: multimodalOCRForm.enabled,
      provider: multimodalOCRForm.provider,
      model: multimodalOCRForm.model.trim() || selectedOCRProvider.default_model,
      api_base:
        multimodalOCRForm.apiBase.trim() || selectedOCRProvider.default_api_base,
      max_pages_per_doc: Math.round(maxPagesPerDoc),
      strategy: multimodalOCRForm.strategy,
      prompt: multimodalOCRForm.prompt.trim(),
    };

    const candidateApiKey = multimodalOCRForm.apiKeyInput.trim();
    if (candidateApiKey.length > 0) {
      payload.api_key = candidateApiKey;
    } else if (clearMultimodalOCRKeyOnSave) {
      payload.api_key = "";
    }
    return payload;
  }, [
    clearMultimodalOCRKeyOnSave,
    multimodalOCRForm,
    ocrProviderOptions,
  ]);

  const saveMultimodalOCRSettings = useCallback(async () => {
    setSavingMultimodalOCR(true);
    setMultimodalOCRError(null);
    setMultimodalOCRNotice(null);
    try {
      const updated = await updateMultimodalOCRSettings(
        buildMultimodalOCRPayload(),
      );
      setMultimodalOCRForm(
        mapMultimodalOCRSettingsToForm(updated, ocrProviderOptions),
      );
      setClearMultimodalOCRKeyOnSave(false);
      setMultimodalOCRNotice("Multimodal OCR settings saved.");
    } catch (error) {
      const reason = error instanceof Error ? error.message : "Unknown error";
      setMultimodalOCRError(
        `Failed to save multimodal OCR settings: ${reason}`,
      );
    } finally {
      setSavingMultimodalOCR(false);
    }
  }, [buildMultimodalOCRPayload, ocrProviderOptions]);

  const updatedAtLabel = useMemo(() => {
    if (!assistantForm.updatedAt) return "Not saved yet";
    const timestamp = Date.parse(assistantForm.updatedAt);
    if (Number.isNaN(timestamp)) return assistantForm.updatedAt;
    return new Date(timestamp).toLocaleString();
  }, [assistantForm.updatedAt]);

  const llamaParseUpdatedAtLabel = useMemo(() => {
    if (!llamaParseForm.updatedAt) return "Not saved yet";
    const timestamp = Date.parse(llamaParseForm.updatedAt);
    if (Number.isNaN(timestamp)) return llamaParseForm.updatedAt;
    return new Date(timestamp).toLocaleString();
  }, [llamaParseForm.updatedAt]);

  const multimodalOCRUpdatedAtLabel = useMemo(() => {
    if (!multimodalOCRForm.updatedAt) return "Not saved yet";
    const timestamp = Date.parse(multimodalOCRForm.updatedAt);
    if (Number.isNaN(timestamp)) return multimodalOCRForm.updatedAt;
    return new Date(timestamp).toLocaleString();
  }, [multimodalOCRForm.updatedAt]);

  const selectedProvider = useMemo(
    () =>
      providerOptions.find((option) => option.provider === assistantForm.llmProvider) ??
      providerOptions[0] ??
      null,
    [assistantForm.llmProvider, providerOptions],
  );
  const selectedEmbeddingProvider = useMemo(
    () =>
      embeddingProviderOptions.find(
        (option) => option.provider === assistantForm.embeddingProvider,
      ) ??
      embeddingProviderOptions[0] ??
      null,
    [assistantForm.embeddingProvider, embeddingProviderOptions],
  );
  const selectedOCRProvider = useMemo(
    () =>
      ocrProviderOptions.find(
        (option) => option.provider === multimodalOCRForm.provider,
      ) ??
      ocrProviderOptions[0] ??
      null,
    [multimodalOCRForm.provider, ocrProviderOptions],
  );
  const retrievalStatusSummary = useMemo(() => {
    if (!retrievalStatus) {
      return "Retrieval pipeline status is unavailable.";
    }
    if (
      retrievalStatus.mode === "recursive-rag" ||
      retrievalStatus.mode === "notebook-recursive"
    ) {
      return "Mode: recursive-rag (aligned with Jupyter pipeline).";
    }
    const reasonText = retrievalStatus.reasons
      .map((reason) => {
        switch (reason) {
          case "no_ready_documents":
            return "no ready documents";
          case "llm_api_key_missing":
            return "LLM API key missing";
          case "embedding_provider_unsupported":
            return "embedding provider unsupported";
          case "embedding_api_key_missing":
            return "embedding API key missing";
          case "embedding_api_key_invalid":
            return "embedding API key invalid";
          default:
            return reason;
        }
      })
      .join(", ");
    return `Mode: fallback (${reasonText || "unknown reason"}).`;
  }, [retrievalStatus]);

  if (!open) {
    return null;
  }

  const isDark = theme === "dark";
  const panelTone = isDark
    ? "border-[#2b3342] bg-[#111723] text-[#e5ebf5]"
    : "border-[#d9dee8] bg-[#f5f7fb] text-[#1a2234]";
  const sidebarTone = isDark
    ? "border-[#273040] bg-[#141b29]"
    : "border-[#d7dce6] bg-[#f1f4f9]";
  const contentTone = isDark ? "bg-[#121a28]" : "bg-[#f8f9fd]";
  const borderTone = isDark ? "border-[#2d384d]" : "border-[#dde2eb]";
  const headingTone = isDark ? "text-[#f3f7ff]" : "text-[#24344d]";
  const labelTone = isDark ? "text-[#9ba8bc]" : "text-[#66758d]";
  const itemTone = isDark
    ? "text-[#c5d1e2] hover:bg-[#1c2739]"
    : "text-[#33465f] hover:bg-[#e8edf6]";
  const activeItemTone = isDark
    ? "bg-[#202c40] text-[#f3f7ff]"
    : "bg-[#e4e9f2] text-[#1f3048]";
  const closeButtonTone = isDark
    ? "text-[#9ba8bc] hover:bg-[#1f2a3d] hover:text-[#f3f7ff]"
    : "text-[#66758d] hover:bg-[#e9edf5] hover:text-[#1a2234]";

  const inputTone = isDark
    ? "border-[#37445a] bg-[#121b2b] text-[#eaf0ff] placeholder:text-[#7f8ba1] focus:ring-[#4A90D9]"
    : "border-[#c8d2e0] bg-white text-[#24344d] placeholder:text-[#8b97ab] focus:ring-[#4A90D9]";
  const selectContentTone = isDark
    ? "border-[#37445a] bg-[#121b2b] text-[#eaf0ff]"
    : "border-[#c8d2e0] bg-white text-[#24344d]";
  const checkboxTone = isDark ? "border-[#4a5a73]" : "border-[#9cb0ca]";
  const checkboxRowTone = isDark ? "hover:bg-[#1a2537]" : "hover:bg-[#edf2f8]";

  return (
    <div
      className={`absolute inset-0 z-[70] flex bg-black/55 ${
        isMobileViewport ? "items-stretch justify-stretch p-0" : "items-center justify-center p-5"
      }`}
      onClick={onClose}
    >
      <div
        className={
          isMobileViewport
            ? `h-full w-full overflow-hidden ${panelTone}`
            : `flex h-[min(820px,92vh)] w-[min(1260px,95vw)] overflow-hidden rounded-2xl border shadow-[0_28px_72px_rgba(0,0,0,0.35)] ${panelTone}`
        }
        onClick={(event) => event.stopPropagation()}
      >
        <div className={`h-full w-full ${isMobileViewport ? "flex flex-col" : "flex"}`}>
          <aside
            className={
              isMobileViewport
                ? `shrink-0 border-b px-5 py-5 ${sidebarTone} ${borderTone}`
                : `w-[270px] shrink-0 border-r px-4 py-5 ${sidebarTone} ${borderTone}`
            }
          >
            <h3 className={`px-3 text-[18px] font-semibold ${headingTone}`}>Workspace</h3>
            <nav className="mt-3 space-y-1">
              <button
                className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-[14px] font-medium ${
                  activeTab === "assistant" ? activeItemTone : itemTone
                }`}
                type="button"
                onClick={() => setActiveTab("assistant")}
              >
                <SettingsIcon size={16} />
                <span>Assistant</span>
              </button>
              <button
                className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-[14px] font-medium ${
                  activeTab === "pdf-table-parse" ? activeItemTone : itemTone
                }`}
                type="button"
                onClick={() => setActiveTab("pdf-table-parse")}
              >
                <Shield size={16} />
                <span>PDF Table Parse</span>
              </button>
              <button
                className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-[14px] font-medium ${itemTone}`}
                type="button"
              >
                <span className="flex items-center gap-3">
                  <FlaskConical size={16} />
                  <span>Lab</span>
                </span>
                <span
                  className={`rounded-full border px-2 py-0.5 text-xs ${isDark ? "border-[#4a5770] text-[#b5c2d7]" : "border-[#c8d0df] text-[#51617b]"}`}
                >
                  Pro
                </span>
              </button>
              <button
                className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-[14px] font-medium ${itemTone}`}
                type="button"
              >
                <Link2 size={16} />
                <span>Connect</span>
              </button>
              <button
                className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-[14px] font-medium ${itemTone}`}
                type="button"
              >
                <Building2 size={16} />
                <span>Enterprise</span>
              </button>
            </nav>
          </aside>

          <section className={`flex min-h-0 min-w-0 flex-1 flex-col ${contentTone}`}>
            <header className={`flex h-[72px] items-center justify-between border-b px-6 sm:px-8 ${borderTone}`}>
              <h2 className={`text-[20px] font-semibold leading-none ${headingTone}`}>
                {activeTab === "assistant" ? "Assistant Settings" : "PDF Table Parse Settings"}
              </h2>
              <button
                onClick={onClose}
                aria-label="Close settings"
                className={`rounded-lg p-1.5 transition-colors ${closeButtonTone}`}
              >
                <X size={20} />
              </button>
            </header>

            <div className="flex-1 overflow-y-auto px-6 py-6 sm:px-8">
              {activeTab === "assistant" ? (
                <>
                  <p className={`text-[13px] ${labelTone}`}>
                    线上运行只使用这里保存的配置，不读取本地 .env。
                  </p>
                  <p className={`mt-1 text-[12px] ${labelTone}`}>
                    Last updated: <span className={headingTone}>{updatedAtLabel}</span>
                  </p>
                  <p
                    className={`mt-3 rounded-lg border px-3 py-2 text-[12px] ${
                      retrievalStatus?.mode === "recursive-rag" ||
                      retrievalStatus?.mode === "notebook-recursive"
                        ? "border-[#56a37a]/60 bg-[#56a37a]/10 text-[#4f9c73]"
                        : "border-[#cc9a5c]/60 bg-[#cc9a5c]/10 text-[#be8a48]"
                    }`}
                  >
                    {retrievalStatusSummary}
                  </p>

                  <div className="mt-5 space-y-4">
                    <h3 className={`text-[13px] font-semibold ${headingTone}`}>
                      LLM Configuration
                    </h3>
                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        Provider
                      </span>
                      <Select
                        value={assistantForm.llmProvider}
                        disabled={isBusy}
                        onValueChange={(value) => {
                          if (value === assistantForm.llmProvider) {
                            return;
                          }
                          void loadProviderSettings({
                            llmProvider: value,
                            embeddingProvider: assistantForm.embeddingProvider,
                          });
                        }}
                      >
                        <SelectTrigger className={`h-12 text-[1.05rem] ${inputTone}`}>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className={selectContentTone}>
                          {providerOptions.map((option) => (
                            <SelectItem key={option.provider} value={option.provider}>
                              {option.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </label>

                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        Model
                      </span>
                      <Input
                        value={assistantForm.llmModel}
                        onChange={(event) =>
                          setAssistantForm((previous) => ({
                            ...previous,
                            llmModel: event.target.value,
                          }))
                        }
                        className={`h-12 text-[1.05rem] ${inputTone}`}
                        placeholder={selectedProvider?.default_model || "qwen-max"}
                      />
                    </label>

                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        API Base URL
                      </span>
                      <Input
                        value={assistantForm.apiBase}
                        onChange={(event) =>
                          setAssistantForm((previous) => ({
                            ...previous,
                            apiBase: event.target.value,
                          }))
                        }
                        className={`h-12 text-[1.05rem] ${inputTone}`}
                        placeholder={
                          selectedProvider?.default_api_base || "https://api.dashscope.aliyuncs.com"
                        }
                      />
                    </label>

                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        Temperature (0 - 2)
                      </span>
                      <Select
                        value={assistantForm.temperature}
                        onValueChange={(value) =>
                          setAssistantForm((previous) => ({
                            ...previous,
                            temperature: value,
                          }))
                        }
                      >
                        <SelectTrigger className={`h-12 text-[1.05rem] ${inputTone}`}>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className={selectContentTone}>
                          {TEMPERATURE_OPTIONS.map((option) => (
                            <SelectItem key={option} value={option}>
                              {option}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </label>

                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        Max tokens (64 - 4096)
                      </span>
                      <Input
                        type="number"
                        min={64}
                        max={4096}
                        step={1}
                        value={assistantForm.maxTokens}
                        onChange={(event) =>
                          setAssistantForm((previous) => ({
                            ...previous,
                            maxTokens: event.target.value,
                          }))
                        }
                        className={`h-12 text-[1.05rem] ${inputTone}`}
                      />
                    </label>

                    <div className="pt-1">
                      <p className={`text-[12px] ${labelTone}`}>
                        Current key status:{" "}
                        <span className={headingTone}>
                          {assistantForm.apiKeyConfigured
                            ? `Configured (${assistantForm.apiKeyPreview || "masked"})`
                            : "Not configured"}
                        </span>
                      </p>
                    </div>

                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        {(selectedProvider?.label || "Provider") + " API Key"}
                      </span>
                      <Input
                        type="password"
                        value={assistantForm.apiKeyInput}
                        onChange={(event) =>
                          setAssistantForm((previous) => ({
                            ...previous,
                            apiKeyInput: event.target.value,
                          }))
                        }
                        placeholder="Leave blank to keep unchanged"
                        className={`h-12 text-[1.05rem] ${inputTone}`}
                      />
                    </label>

                    <label
                      htmlFor="clear-api-key-on-save"
                      className={`flex items-center gap-2 rounded-md px-1 py-1 transition-colors ${checkboxRowTone}`}
                    >
                      <Checkbox
                        id="clear-api-key-on-save"
                        checked={clearApiKeyOnSave}
                        onCheckedChange={(checked) => setClearApiKeyOnSave(checked === true)}
                        disabled={assistantForm.apiKeyInput.trim().length > 0}
                        className={checkboxTone}
                      />
                      <span className={`text-[12px] leading-none ${labelTone}`}>
                        Clear existing key on save
                      </span>
                    </label>

                    <div className={`my-2 border-t ${borderTone}`} />

                    <h3 className={`text-[13px] font-semibold ${headingTone}`}>
                      Embedding Configuration
                    </h3>

                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        Embedding Provider
                      </span>
                      <Select
                        value={assistantForm.embeddingProvider}
                        disabled={isBusy}
                        onValueChange={(value) => {
                          if (value === assistantForm.embeddingProvider) {
                            return;
                          }
                          void loadProviderSettings({
                            llmProvider: assistantForm.llmProvider,
                            embeddingProvider: value,
                          });
                        }}
                      >
                        <SelectTrigger className={`h-12 text-[1.05rem] ${inputTone}`}>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className={selectContentTone}>
                          {embeddingProviderOptions.map((option) => (
                            <SelectItem key={option.provider} value={option.provider}>
                              {option.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </label>

                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        Embedding Model
                      </span>
                      <Input
                        value={assistantForm.embeddingModel}
                        onChange={(event) =>
                          setAssistantForm((previous) => ({
                            ...previous,
                            embeddingModel: event.target.value,
                          }))
                        }
                        className={`h-12 text-[1.05rem] ${inputTone}`}
                        placeholder={
                          selectedEmbeddingProvider?.default_model || "text-embedding-v4"
                        }
                      />
                    </label>

                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        Embedding API Base URL
                      </span>
                      <Input
                        value={assistantForm.embeddingApiBase}
                        onChange={(event) =>
                          setAssistantForm((previous) => ({
                            ...previous,
                            embeddingApiBase: event.target.value,
                          }))
                        }
                        className={`h-12 text-[1.05rem] ${inputTone}`}
                        placeholder={
                          selectedEmbeddingProvider?.default_api_base ||
                          "https://api.dashscope.aliyuncs.com"
                        }
                      />
                    </label>

                    <div className="pt-1">
                      <p className={`text-[12px] ${labelTone}`}>
                        Embedding key status:{" "}
                        <span className={headingTone}>
                          {assistantForm.embeddingApiKeyConfigured
                            ? `Configured (${assistantForm.embeddingApiKeyPreview || "masked"})`
                            : "Not configured"}
                        </span>
                      </p>
                    </div>

                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        {(selectedEmbeddingProvider?.label || "Embedding provider") + " API Key"}
                      </span>
                      <Input
                        type="password"
                        value={assistantForm.embeddingApiKeyInput}
                        onChange={(event) =>
                          setAssistantForm((previous) => ({
                            ...previous,
                            embeddingApiKeyInput: event.target.value,
                          }))
                        }
                        placeholder="Leave blank to keep unchanged"
                        className={`h-12 text-[1.05rem] ${inputTone}`}
                      />
                    </label>

                    <label
                      htmlFor="clear-embedding-api-key-on-save"
                      className={`flex items-center gap-2 rounded-md px-1 py-1 transition-colors ${checkboxRowTone}`}
                    >
                      <Checkbox
                        id="clear-embedding-api-key-on-save"
                        checked={clearEmbeddingApiKeyOnSave}
                        onCheckedChange={(checked) =>
                          setClearEmbeddingApiKeyOnSave(checked === true)
                        }
                        disabled={assistantForm.embeddingApiKeyInput.trim().length > 0}
                        className={checkboxTone}
                      />
                      <span className={`text-[12px] leading-none ${labelTone}`}>
                        Clear embedding key on save
                      </span>
                    </label>
                  </div>

                  {assistantError && (
                    <p className="mt-4 rounded-lg border border-[#cc5c67]/60 bg-[#cc5c67]/10 px-3 py-2 text-sm text-[#d86873]">
                      {assistantError}
                    </p>
                  )}
                  {assistantNotice && (
                    <p className="mt-4 rounded-lg border border-[#56a37a]/60 bg-[#56a37a]/10 px-3 py-2 text-sm text-[#4f9c73]">
                      {assistantNotice}
                    </p>
                  )}

                  <div className="mt-5 flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      onClick={() => void loadAssistantSettings()}
                      disabled={isBusy}
                      className={`rounded-lg border px-4 py-2 text-sm font-semibold transition-colors ${borderTone} ${headingTone} disabled:cursor-not-allowed disabled:opacity-65`}
                    >
                      {loadingAssistant || loadingLlamaParse ? "Loading..." : "Reload"}
                    </button>
                    <button
                      type="button"
                      onClick={() => void runAssistantConnectionTest()}
                      disabled={isBusy}
                      className="rounded-lg border border-[#4A90D9]/55 px-4 py-2 text-sm font-semibold text-[#4A90D9] transition-colors hover:bg-[#4A90D9]/10 disabled:cursor-not-allowed disabled:opacity-65"
                    >
                      {testingAssistant ? "Testing..." : "Test Connection"}
                    </button>
                    <button
                      type="button"
                      onClick={() => void saveAssistantSettings()}
                      disabled={isBusy}
                      className="inline-flex items-center gap-2 rounded-lg bg-[#4A90D9] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#3f82c8] disabled:cursor-not-allowed disabled:opacity-65"
                    >
                      <Save size={14} />
                      {savingAssistant ? "Saving..." : "Save Assistant Settings"}
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <p className={`text-[13px] ${labelTone}`}>
                    该配置用于 PDF 表格解析，和 Assistant provider 凭据分开管理。
                  </p>
                  <p className={`mt-1 text-[12px] ${labelTone}`}>
                    LlamaParse updated:{" "}
                    <span className={headingTone}>{llamaParseUpdatedAtLabel}</span>
                  </p>
                  <p className={`mt-1 text-[12px] ${labelTone}`}>
                    Multimodal OCR updated:{" "}
                    <span className={headingTone}>{multimodalOCRUpdatedAtLabel}</span>
                  </p>

                  <div className="mt-5 space-y-4">
                    <label
                      htmlFor="llamaparse-enabled"
                      className={`flex items-center gap-2 rounded-md px-1 py-1 transition-colors ${checkboxRowTone}`}
                    >
                      <Checkbox
                        id="llamaparse-enabled"
                        checked={llamaParseForm.enabled}
                        onCheckedChange={(checked) =>
                          setLlamaParseForm((previous) => ({
                            ...previous,
                            enabled: checked === true,
                          }))
                        }
                        className={checkboxTone}
                      />
                      <span className={`text-[12px] leading-none ${labelTone}`}>
                        Enable LlamaParse for PDF table parsing
                      </span>
                    </label>

                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        API Base URL
                      </span>
                      <Input
                        value={llamaParseForm.baseUrl}
                        onChange={(event) =>
                          setLlamaParseForm((previous) => ({
                            ...previous,
                            baseUrl: event.target.value,
                          }))
                        }
                        className={`h-12 text-[1.05rem] ${inputTone}`}
                        placeholder="https://api.cloud.llamaindex.ai"
                      />
                    </label>

                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        Parse Pages (optional)
                      </span>
                      <Input
                        value={llamaParseForm.targetPages}
                        onChange={(event) =>
                          setLlamaParseForm((previous) => ({
                            ...previous,
                            targetPages: event.target.value,
                          }))
                        }
                        className={`h-12 text-[1.05rem] ${inputTone}`}
                        placeholder="Examples: 1,2,3 or 1-3,8"
                      />
                    </label>

                    <label
                      htmlFor="llamaparse-split-by-page"
                      className={`flex items-center gap-2 rounded-md px-1 py-1 transition-colors ${checkboxRowTone}`}
                    >
                      <Checkbox
                        id="llamaparse-split-by-page"
                        checked={llamaParseForm.splitByPage}
                        onCheckedChange={(checked) =>
                          setLlamaParseForm((previous) => ({
                            ...previous,
                            splitByPage: checked === true,
                          }))
                        }
                        className={checkboxTone}
                      />
                      <span className={`text-[12px] leading-none ${labelTone}`}>
                        Split markdown output by page
                      </span>
                    </label>

                    <div className="pt-1">
                      <p className={`text-[12px] ${labelTone}`}>
                        Current key status:{" "}
                        <span className={headingTone}>
                          {llamaParseForm.apiKeyConfigured
                            ? `Configured (${llamaParseForm.apiKeyPreview || "masked"})`
                            : "Not configured"}
                        </span>
                      </p>
                    </div>

                    <label className="block">
                      <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                        LlamaParse API Key
                      </span>
                      <Input
                        type="password"
                        value={llamaParseForm.apiKeyInput}
                        onChange={(event) =>
                          setLlamaParseForm((previous) => ({
                            ...previous,
                            apiKeyInput: event.target.value,
                          }))
                        }
                        placeholder="Leave blank to keep unchanged"
                        className={`h-12 text-[1.05rem] ${inputTone}`}
                      />
                    </label>

                    <label
                      htmlFor="clear-llamaparse-api-key-on-save"
                      className={`flex items-center gap-2 rounded-md px-1 py-1 transition-colors ${checkboxRowTone}`}
                    >
                      <Checkbox
                        id="clear-llamaparse-api-key-on-save"
                        checked={clearLlamaParseKeyOnSave}
                        onCheckedChange={(checked) => setClearLlamaParseKeyOnSave(checked === true)}
                        disabled={llamaParseForm.apiKeyInput.trim().length > 0}
                        className={checkboxTone}
                      />
                      <span className={`text-[12px] leading-none ${labelTone}`}>
                        Clear existing key on save
                      </span>
                    </label>

                    <p className={`text-[12px] ${labelTone}`}>
                      该能力会产生 LlamaParse 外部调用成本。建议先用 `target_pages` 做小范围测试。
                    </p>
                  </div>

                  <div className={`mt-6 border-t pt-5 ${borderTone}`}>
                    <h3 className={`text-[13px] font-semibold ${headingTone}`}>
                      Multimodal OCR (Vision LLM)
                    </h3>
                    <p className={`mt-1 text-[12px] ${labelTone}`}>
                      用于补强 LlamaParse 在扫描件、复杂表格、低质量 PDF 场景下的解析质量。
                    </p>

                    <div className="mt-4 space-y-4">
                      <label
                        htmlFor="multimodal-ocr-enabled"
                        className={`flex items-center gap-2 rounded-md px-1 py-1 transition-colors ${checkboxRowTone}`}
                      >
                        <Checkbox
                          id="multimodal-ocr-enabled"
                          checked={multimodalOCRForm.enabled}
                          onCheckedChange={(checked) =>
                            setMultimodalOCRForm((previous) => ({
                              ...previous,
                              enabled: checked === true,
                            }))
                          }
                          className={checkboxTone}
                        />
                        <span className={`text-[12px] leading-none ${labelTone}`}>
                          Enable multimodal OCR for PDF parsing
                        </span>
                      </label>

                      <label className="block">
                        <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                          OCR Provider
                        </span>
                        <Select
                          value={multimodalOCRForm.provider}
                          disabled={isBusy}
                          onValueChange={(value) =>
                            setMultimodalOCRForm((previous) => {
                              const selected =
                                ocrProviderOptions.find(
                                  (option) => option.provider === value,
                                ) ?? null;
                              return {
                                ...previous,
                                provider: value,
                                model:
                                  previous.model.trim() ||
                                  selected?.default_model ||
                                  previous.model,
                                apiBase:
                                  previous.apiBase.trim() ||
                                  selected?.default_api_base ||
                                  previous.apiBase,
                              };
                            })
                          }
                        >
                          <SelectTrigger className={`h-12 text-[1.05rem] ${inputTone}`}>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent className={selectContentTone}>
                            {ocrProviderOptions.map((option) => (
                              <SelectItem key={option.provider} value={option.provider}>
                                {option.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </label>

                      <label className="block">
                        <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                          OCR Model
                        </span>
                        <Input
                          value={multimodalOCRForm.model}
                          onChange={(event) =>
                            setMultimodalOCRForm((previous) => ({
                              ...previous,
                              model: event.target.value,
                            }))
                          }
                          className={`h-12 text-[1.05rem] ${inputTone}`}
                          placeholder={selectedOCRProvider?.default_model || "gpt-4o-mini"}
                        />
                      </label>

                      <label className="block">
                        <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                          OCR API Base URL
                        </span>
                        <Input
                          value={multimodalOCRForm.apiBase}
                          onChange={(event) =>
                            setMultimodalOCRForm((previous) => ({
                              ...previous,
                              apiBase: event.target.value,
                            }))
                          }
                          className={`h-12 text-[1.05rem] ${inputTone}`}
                          placeholder={
                            selectedOCRProvider?.default_api_base || "https://api.openai.com/v1"
                          }
                        />
                      </label>

                      <label className="block">
                        <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                          Max pages per document (1-128)
                        </span>
                        <Input
                          type="number"
                          min={1}
                          max={128}
                          step={1}
                          value={multimodalOCRForm.maxPagesPerDoc}
                          onChange={(event) =>
                            setMultimodalOCRForm((previous) => ({
                              ...previous,
                              maxPagesPerDoc: event.target.value,
                            }))
                          }
                          className={`h-12 text-[1.05rem] ${inputTone}`}
                          placeholder="8"
                        />
                      </label>

                      <label className="block">
                        <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                          Merge strategy
                        </span>
                        <Select
                          value={multimodalOCRForm.strategy}
                          onValueChange={(value) =>
                            setMultimodalOCRForm((previous) => ({
                              ...previous,
                              strategy: value as "fallback" | "prefer_ocr" | "hybrid",
                            }))
                          }
                        >
                          <SelectTrigger className={`h-12 text-[1.05rem] ${inputTone}`}>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent className={selectContentTone}>
                            <SelectItem value="fallback">fallback</SelectItem>
                            <SelectItem value="hybrid">hybrid</SelectItem>
                            <SelectItem value="prefer_ocr">prefer_ocr</SelectItem>
                          </SelectContent>
                        </Select>
                      </label>

                      <div className="pt-1">
                        <p className={`text-[12px] ${labelTone}`}>
                          OCR key status:{" "}
                          <span className={headingTone}>
                            {multimodalOCRForm.apiKeyConfigured
                              ? `Configured (${multimodalOCRForm.apiKeyPreview || "masked"})`
                              : "Not configured"}
                          </span>
                        </p>
                      </div>

                      <label className="block">
                        <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                          OCR API Key
                        </span>
                        <Input
                          type="password"
                          value={multimodalOCRForm.apiKeyInput}
                          onChange={(event) =>
                            setMultimodalOCRForm((previous) => ({
                              ...previous,
                              apiKeyInput: event.target.value,
                            }))
                          }
                          placeholder="Leave blank to keep unchanged"
                          className={`h-12 text-[1.05rem] ${inputTone}`}
                        />
                      </label>

                      <label
                        htmlFor="clear-multimodal-ocr-api-key-on-save"
                        className={`flex items-center gap-2 rounded-md px-1 py-1 transition-colors ${checkboxRowTone}`}
                      >
                        <Checkbox
                          id="clear-multimodal-ocr-api-key-on-save"
                          checked={clearMultimodalOCRKeyOnSave}
                          onCheckedChange={(checked) =>
                            setClearMultimodalOCRKeyOnSave(checked === true)
                          }
                          disabled={multimodalOCRForm.apiKeyInput.trim().length > 0}
                          className={checkboxTone}
                        />
                        <span className={`text-[12px] leading-none ${labelTone}`}>
                          Clear existing OCR key on save
                        </span>
                      </label>

                      <label className="block">
                        <span className={`mb-1.5 block text-[12px] font-semibold ${labelTone}`}>
                          OCR Prompt (optional)
                        </span>
                        <textarea
                          value={multimodalOCRForm.prompt}
                          onChange={(event) =>
                            setMultimodalOCRForm((previous) => ({
                              ...previous,
                              prompt: event.target.value,
                            }))
                          }
                          placeholder="Leave empty to use built-in OCR extraction prompt."
                          className={`min-h-[110px] w-full rounded-md border px-3 py-2 text-sm ${inputTone}`}
                        />
                      </label>

                      <p className={`text-[12px] ${labelTone}`}>
                        建议先用 `hybrid` 策略。若是扫描件或复杂表格，可切换到 `prefer_ocr`。
                      </p>
                    </div>
                  </div>

                  {llamaParseError && (
                    <p className="mt-4 rounded-lg border border-[#cc5c67]/60 bg-[#cc5c67]/10 px-3 py-2 text-sm text-[#d86873]">
                      {llamaParseError}
                    </p>
                  )}
                  {llamaParseNotice && (
                    <p className="mt-4 rounded-lg border border-[#56a37a]/60 bg-[#56a37a]/10 px-3 py-2 text-sm text-[#4f9c73]">
                      {llamaParseNotice}
                    </p>
                  )}
                  {multimodalOCRError && (
                    <p className="mt-4 rounded-lg border border-[#cc5c67]/60 bg-[#cc5c67]/10 px-3 py-2 text-sm text-[#d86873]">
                      {multimodalOCRError}
                    </p>
                  )}
                  {multimodalOCRNotice && (
                    <p className="mt-4 rounded-lg border border-[#56a37a]/60 bg-[#56a37a]/10 px-3 py-2 text-sm text-[#4f9c73]">
                      {multimodalOCRNotice}
                    </p>
                  )}

                  <div className="mt-5 flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      onClick={() => void loadAssistantSettings()}
                      disabled={isBusy}
                      className={`rounded-lg border px-4 py-2 text-sm font-semibold transition-colors ${borderTone} ${headingTone} disabled:cursor-not-allowed disabled:opacity-65`}
                    >
                      {loadingAssistant || loadingLlamaParse || loadingMultimodalOCR
                        ? "Loading..."
                        : "Reload"}
                    </button>
                    <button
                      type="button"
                      onClick={() => void saveLlamaParseSettings()}
                      disabled={isBusy}
                      className="inline-flex items-center gap-2 rounded-lg bg-[#2c8f6f] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#267e62] disabled:cursor-not-allowed disabled:opacity-65"
                    >
                      <Save size={14} />
                      {savingLlamaParse ? "Saving..." : "Save LlamaParse Settings"}
                    </button>
                    <button
                      type="button"
                      onClick={() => void saveMultimodalOCRSettings()}
                      disabled={isBusy}
                      className="inline-flex items-center gap-2 rounded-lg bg-[#4A90D9] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#3f82c8] disabled:cursor-not-allowed disabled:opacity-65"
                    >
                      <Save size={14} />
                      {savingMultimodalOCR ? "Saving..." : "Save Multimodal OCR Settings"}
                    </button>
                  </div>
                </>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
