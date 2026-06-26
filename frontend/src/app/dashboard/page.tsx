"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  FlaskConical,
  History,
  LogOut,
  Loader2,
  AlertCircle,
  CheckCircle2,
  ArrowRight,
  Clock,
  User,
} from "lucide-react";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { auth } from "@/lib/firebase";
import { ThemeToggle } from "@/components/ThemeToggle";
import type { PredictionResult } from "@/types";

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(true);

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, async (u) => {
      setUser(u);
      setLoading(false);
      if (!u) {
        router.push("/login");
        return;
      }
      try {
        const token = await u.getIdToken();
        const res = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860"}/api/history`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (res.ok) {
          const data = await res.json();
          setPredictions(data.predictions || []);
        }
      } catch {
        // silently fail
      } finally {
        setLoadingHistory(false);
      }
    });
    return unsub;
  }, [router]);

  const handleSignOut = async () => {
    await signOut(auth);
    router.push("/");
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
          <Link
            href="/predict"
            className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground"
          >
            New Prediction
          </Link>
          <button
            onClick={handleSignOut}
            className="rounded-lg border border-border px-3 py-2 text-sm"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </nav>

      <main className="relative z-10 flex-1 px-6 py-8 sm:px-8">
        <div className="mx-auto max-w-4xl">
          <div className="mb-6">
            <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              View your prediction history and saved results.
            </p>
          </div>

          {loadingHistory ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
            </div>
          ) : predictions.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-xl border border-border bg-card p-12 text-center"
            >
              <History className="mx-auto h-12 w-12 text-muted-foreground/40" />
              <h2 className="mt-4 text-lg font-semibold">No predictions yet</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Start by predicting a variant effect.
              </p>
              <Link
                href="/predict"
                className="mt-6 inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground"
              >
                Predict a Variant
                <ArrowRight className="h-4 w-4" />
              </Link>
            </motion.div>
          ) : (
            <div className="space-y-3">
              {predictions.map((pred, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="flex items-center justify-between rounded-xl border border-border bg-card p-4"
                >
                  <div className="flex items-center gap-3">
                    {pred.prediction === "pathogenic" ? (
                      <AlertCircle className="h-5 w-5 text-red-500" />
                    ) : (
                      <CheckCircle2 className="h-5 w-5 text-green-500" />
                    )}
                    <div>
                      <div className="text-sm font-medium">{pred.variant}</div>
                      <div className="text-xs text-muted-foreground">
                        {pred.model_type} &middot; {(pred.confidence * 100).toFixed(0)}% confidence
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <div className="text-sm font-medium capitalize">{pred.prediction}</div>
                      <div className="text-xs text-muted-foreground">
                        <Clock className="mr-0.5 inline h-3 w-3" />
                        {pred.inference_time_ms.toFixed(0)}ms
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
