"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  FlaskConical,
  Dna,
  Activity,
  ArrowRight,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Clock,
  Beaker,
} from "lucide-react";
import { onAuthStateChanged } from "firebase/auth";
import { auth } from "@/lib/firebase";
import { predictVariant } from "@/lib/api";
import type { PredictionResult } from "@/types";
import { ThemeToggle } from "@/components/ThemeToggle";

const EXAMPLE_SEQUENCES = [
  { label: "TP53 (p53 tumor suppressor)", seq: "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNT" },
  { label: "BRCA1 (breast cancer type 1)", seq: "MDLSALRVEEVQNVINAMQKILECPICLELLIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSIIQSMGYRNRAKRLLQSEPENPSLQETSLSVQLSNLGTVRTLRTKQRIQPQKTSVYIELGSDSSEDTVNKATYCSVGDQELLQITPQGTRDEISLDSAKKAACEFSETDVTNTEHHQPSNNDLNTTEKRAAERHPEKYQGSSVSNLHVEPCGTNTHASSLQHENSSLLLTKDRMNVEKAEFCNKSKQPGLARSQHNRWAGSKETCNDRRTPSTEKKVDLNADPLCERKEWNKQKLPCSENPRDTEDVPWITLNSSIQKVNEWFSRSDELLGSDDSHDGESESNAKVADVLDVLNEVDEYSGSSEKIDLLASDPHEALICKSERVHSKSVESNIEDKIFGKTYRKKASLPNLSHISENSLVNQSIELSRIRESGGLSSLGHSEPVQPSNEEESILSKCSSDEGKELKLGTEAHGSGESPIESGEGPSTSDTLSVSEDQELADSTSEEEEEADTSSDDPELMTDSEHEETRKILTYTMEDQKATGRIETNITEHSAEDSSRSDQENELQDLTFSVSETQSSRKSFGEFGKRSNSKKGNESETGSESEGSQVSSNPSVCSELDLTELPEGSDSRESTLAVDDSSGENDRSLTSEASGSPQKSSHSPNSDSTSLSEISRLEMPTSSPSQSPAPDWNVTTLSSISSSSSLGDPQNLFHIKAVGGIKESELLQKVSADTWTRSVTSQMNKILENQGNTGRSKDERRNDQEKNGGDKSQDGKIQEPSQESCEGIPKLVSESETSNIESVDANGSPVSQD" },
  { label: "EGFR (epidermal growth factor receptor)", seq: "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFM" },
];

export default function PredictPage() {
  const router = useRouter();
  const [user, setUser] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [sequence, setSequence] = useState("");
  const [modelType, setModelType] = useState("esm2");
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, (u) => {
      setUser(u);
      setLoading(false);
    });
    return unsub;
  }, []);

  const handlePredict = async () => {
    if (!sequence.trim()) return;
    setPredicting(true);
    setError("");
    setResult(null);
    try {
      const res = await predictVariant(sequence.trim(), modelType);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setPredicting(false);
    }
  };

  const loadExample = (seq: string) => {
    setSequence(seq);
    setResult(null);
    setError("");
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="relative flex min-h-screen flex-col">
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-br from-gradient-start via-gradient-mid to-gradient-end" />
      </div>

      <nav className="relative z-10 flex items-center justify-between border-b border-border px-6 py-4 sm:px-8">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
            <FlaskConical className="h-4 w-4 text-primary-foreground" />
          </div>
          <span className="text-lg font-bold tracking-tight">VariantScope</span>
        </Link>
        <div className="flex items-center gap-3">
          <ThemeToggle />
          {user ? (
            <Link
              href="/dashboard"
              className="rounded-lg border border-border px-4 py-2 text-sm font-medium"
            >
              Dashboard
            </Link>
          ) : (
            <Link
              href="/login"
              className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground"
            >
              Sign In
            </Link>
          )}
        </div>
      </nav>

      <main className="relative z-10 flex flex-1 px-6 py-8 sm:px-8">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 lg:flex-row">
          <div className="flex-1 space-y-6">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Variant Effect Prediction</h1>
              <p className="mt-1 text-sm text-muted-foreground">
                Enter a protein or DNA sequence to predict variant pathogenicity using our dual-engine framework.
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground">Model Engine</label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring"
              >
                <option value="esm2">ESM-2 (Protein Engine)</option>
                <option value="dnabert2">DNABERT-2 (Genomic Engine)</option>
                <option value="hybrid">Hybrid (ESM-2 + DNABERT-2)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground">Sequence</label>
              <textarea
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                rows={8}
                className="mt-1 block w-full rounded-lg border border-border bg-background px-3 py-2 font-mono text-xs leading-relaxed focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring"
                placeholder="Paste your protein or DNA sequence here..."
              />
            </div>

            <div className="space-y-2">
              <p className="text-xs font-medium text-muted-foreground">Example sequences:</p>
              <div className="flex flex-wrap gap-2">
                {EXAMPLE_SEQUENCES.map((ex) => (
                  <button
                    key={ex.label}
                    onClick={() => loadExample(ex.seq)}
                    className="rounded-lg border border-border bg-secondary/50 px-3 py-1.5 text-xs font-medium transition-all hover:border-primary/30 hover:bg-primary/5"
                  >
                    {ex.label}
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={predicting || !sequence.trim()}
              className="flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90 disabled:opacity-50"
            >
              {predicting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <ArrowRight className="h-4 w-4" />
              )}
              Predict Pathogenicity
            </button>

            {error && (
              <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                <AlertCircle className="h-4 w-4 shrink-0" />
                {error}
              </div>
            )}
          </div>

          <div className="w-full lg:w-96">
            {predicting ? (
              <div className="flex h-64 items-center justify-center rounded-xl border border-border bg-card">
                <div className="text-center">
                  <Loader2 className="mx-auto h-8 w-8 animate-spin text-primary" />
                  <p className="mt-3 text-sm text-muted-foreground">Running inference...</p>
                </div>
              </div>
            ) : result ? (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="space-y-4"
              >
                <div className="rounded-xl border border-border bg-card p-6">
                  <h2 className="mb-4 text-base font-semibold">Prediction Result</h2>

                  <div className="mb-4 flex items-center gap-3">
                    {result.prediction === "pathogenic" ? (
                      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-red-100">
                        <AlertCircle className="h-6 w-6 text-red-600" />
                      </div>
                    ) : (
                      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100">
                        <CheckCircle2 className="h-6 w-6 text-green-600" />
                      </div>
                    )}
                    <div>
                      <div className="text-lg font-bold capitalize">{result.prediction}</div>
                      <div className="text-xs text-muted-foreground">
                        {result.model_type === "esm2"
                          ? "ESM-2 Protein Engine"
                          : result.model_type === "dnabert2"
                          ? "DNABERT-2 Genomic Engine"
                          : "Hybrid ESM-2 + DNABERT-2"}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Pathogenicity Score</span>
                        <span className="font-mono font-medium">
                          {result.pathogenicity_score.toFixed(4)}
                        </span>
                      </div>
                      <div className="mt-1 h-2 rounded-full bg-secondary">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${result.pathogenicity_score * 100}%`,
                            backgroundColor:
                              result.pathogenicity_score > 0.7
                                ? "#dc2626"
                                : result.pathogenicity_score > 0.5
                                ? "#f59e0b"
                                : "#22c55e",
                          }}
                        />
                      </div>
                    </div>

                    <div className="flex items-center justify-between border-t border-border pt-3 text-xs">
                      <span className="flex items-center gap-1 text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        Inference Time
                      </span>
                      <span className="font-mono">{result.inference_time_ms.toFixed(0)} ms</span>
                    </div>

                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Confidence</span>
                      <span className="font-mono">{(result.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2 rounded-lg border border-border bg-card p-4">
                  <Beaker className="h-4 w-4 text-primary" />
                  <span className="text-xs text-muted-foreground">
                    Feature-level attribution with PLM-SAE disentanglement available in the full report
                  </span>
                </div>
              </motion.div>
            ) : (
              <div className="flex h-64 items-center justify-center rounded-xl border border-border bg-card">
                <div className="text-center">
                  <Dna className="mx-auto h-8 w-8 text-muted-foreground/50" />
                  <p className="mt-3 text-sm text-muted-foreground">
                    Enter a sequence and run prediction to see results
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
