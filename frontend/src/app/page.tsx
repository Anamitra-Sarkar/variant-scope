"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Dna, Microscope, Brain, ArrowRight, FlaskConical, Activity } from "lucide-react";
import { ThemeToggle } from "@/components/ThemeToggle";

const features = [
  {
    icon: Dna,
    title: "Dual-Engine Transformer Architecture",
    description:
      "Coordinates ESM-2 (650M protein language model) and DNABERT-2 (117M genomic transformer) through sparse feature disentanglement and cross-fitness routing.",
  },
  {
    icon: Microscope,
    title: "Sparse Autoencoder Disentanglement",
    description:
      "PLM-SAEs isolate biophysically interpretable features from dense transformer embeddings, enabling targeted pathogenic signal amplification and noise suppression.",
  },
  {
    icon: Brain,
    title: "Low-Resource Fine-Tuning",
    description:
      "Four parameter-efficient strategies — LoRA, BitFit, Inductive Bias Head, and selective layer unfreezing — optimized for regimes with fewer than 500 annotated variants.",
  },
];

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.12, delayChildren: 0.2 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" as const } },
};

export default function LandingPage() {
  return (
    <div className="relative flex min-h-screen flex-col">
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-br from-gradient-start via-gradient-mid to-gradient-end" />
        <div className="absolute left-1/4 top-1/4 h-96 w-96 rounded-full bg-blue-100/60 blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 h-80 w-80 rounded-full bg-purple-100/60 blur-[120px]" />
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
            href="/login"
            className="rounded-lg border border-border bg-secondary/50 px-4 py-2 text-sm font-medium transition-all hover:border-primary/30 hover:bg-primary/5"
          >
            Sign In
          </Link>
          <Link
            href="/login?register=true"
            className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90"
          >
            Get Started
          </Link>
        </div>
      </nav>

      <main className="relative z-10 flex flex-1 flex-col items-center px-6 py-16 sm:py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" as const }}
          className="text-center"
        >
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-4 py-1.5 text-xs font-medium text-primary">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            Research-Grade Variant Effect Prediction
          </div>

          <h1 className="mx-auto max-w-4xl text-4xl font-bold leading-tight tracking-tight sm:text-5xl lg:text-6xl">
            Genomic and Proteomic{" "}
            <span className="text-primary">Variant Effect</span>
            {" "}Prediction Framework
          </h1>

          <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-muted-foreground sm:text-lg">
            A dual-engine transformer optimization framework combining ESM-2 and DNABERT-2
            with sparse autoencoder disentanglement and differentiable cross-fitness routing
            for somatic mutation pathogenicity classification.
          </p>

          <div className="mt-10 flex items-center justify-center gap-4">
            <Link
              href="/predict"
              className="group inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:bg-primary/90 hover:shadow-xl hover:shadow-primary/25"
            >
              Predict Variant
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <Link
              href="/login"
              className="rounded-lg border border-border px-6 py-3 text-sm font-medium text-muted-foreground transition-all hover:border-primary/30 hover:text-foreground"
            >
              View Documentation
            </Link>
          </div>
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="mx-auto mt-20 grid max-w-5xl gap-6 sm:grid-cols-3"
        >
          {features.map((feature) => (
            <motion.div key={feature.title} variants={item}>
              <div className="group relative flex h-full flex-col rounded-xl border border-border bg-card p-6 transition-all duration-200 hover:border-primary/30 hover:shadow-sm">
                <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <feature.icon className="h-5 w-5" />
                </div>
                <h3 className="mb-2 text-base font-semibold">
                  {feature.title}
                </h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {feature.description}
                </p>
              </div>
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mx-auto mt-20 w-full max-w-3xl"
        >
          <div className="rounded-xl border border-border bg-card p-6 sm:p-8">
            <div className="mb-4 flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Model Architecture</h2>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                { name: "ESM-2-650M", role: "Protein Engine" },
                { name: "DNABERT-2", role: "Genomic Engine" },
                { name: "PLM-SAE", role: "Feature Disentanglement" },
                { name: "Cross-Fitness", role: "Pathogenicity Routing" },
              ].map((m) => (
                <div
                  key={m.name}
                  className="rounded-lg border border-border bg-secondary/50 px-3 py-3 text-center"
                >
                  <div className="text-xs font-semibold text-foreground">{m.name}</div>
                  <div className="mt-0.5 text-[10px] text-muted-foreground">{m.role}</div>
                </div>
              ))}
            </div>
            <p className="mt-4 text-xs text-muted-foreground">
              Dual-engine framework with PLM-SAE sparse disentanglement and differentiable
              cross-fitness routing for calibrated pathogenicity prediction
            </p>
          </div>
        </motion.div>
      </main>

      <footer className="relative z-10 border-t border-border px-8 py-6">
        <div className="flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            VariantScope &mdash; Dual-Engine Variant Effect Prediction Framework
          </p>
          <ThemeToggle />
        </div>
      </footer>
    </div>
  );
}
