import { Link } from 'react-router-dom';
import {
  FlaskConical,
  ArrowLeft,
  Database,
  Cpu,
  Network,
  Brain,
  RefreshCw,
  Layers,
  HardDrive,
  Microscope,
  Dna,
  Globe,
  Server,
} from 'lucide-react';
import { Header } from '../components/Header';

function Section({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="card-surface p-6 animate-slide-up">
      <div className="flex items-center gap-3 mb-4">
        {icon}
        <h2 className="text-lg font-semibold text-text-primary">{title}</h2>
      </div>
      <div className="text-sm text-text-secondary leading-relaxed space-y-3">
        {children}
      </div>
    </section>
  );
}

function ArchNode({
  icon,
  label,
  active = false,
}: {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
}) {
  return (
    <div
      className={`flex flex-col items-center gap-2 p-4 rounded-card border ${
        active
          ? 'border-accent/40 bg-accent/5'
          : 'border-navy-600 bg-navy-900/60'
      }`}
    >
      {icon}
      <span
        className={`text-xs font-medium text-center ${
          active ? 'text-accent' : 'text-text-secondary'
        }`}
      >
        {label}
      </span>
    </div>
  );
}

export function AboutPage() {
  return (
    <div className="min-h-screen bg-navy-900">
      <Header />

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Breadcrumb */}
        <nav className="mb-8">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-sm text-text-secondary hover:text-accent transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Discoveries
          </Link>
        </nav>

        {/* Page Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-3">
            <FlaskConical className="w-8 h-8 text-accent" />
            <span className="text-sm font-medium text-accent tracking-wider uppercase">
              Documentation
            </span>
          </div>
          <h1 className="text-3xl md:text-4xl font-bold text-text-primary mb-4">
            About <span className="text-gradient">ASI-Evolve</span>
          </h1>
          <p className="text-base text-text-secondary max-w-2xl leading-relaxed">
            ASI-Evolve is an autonomous molecular discovery platform that combines predictive
            machine learning models, molecular docking simulations, and ADMET property evaluation
            within an iterative agent-based feedback loop.
          </p>
        </div>

        {/* Architecture Diagram */}
        <Section
          icon={<Network className="w-5 h-5 text-accent" />}
          title="System Architecture"
        >
          <div className="flex flex-col items-center gap-3 my-6">
            {/* Top Row */}
            <div className="grid grid-cols-3 gap-4 w-full max-w-lg">
              <ArchNode icon={<Database className="w-5 h-5 text-warning" />} label="ChEMBL" />
              <ArchNode icon={<Server className="w-5 h-5 text-warning" />} label="PDB" />
              <ArchNode icon={<Globe className="w-5 h-5 text-warning" />} label="SwissADME" />
            </div>

            {/* Arrows */}
            <div className="text-text-secondary text-xs">Data Sources &darr;</div>

            {/* Middle Row */}
            <div className="grid grid-cols-3 gap-4 w-full max-w-lg">
              <ArchNode icon={<Dna className="w-5 h-5 text-accent" />} label="Fingerprint" active />
              <ArchNode icon={<Brain className="w-5 h-5 text-accent" />} label="RandomForest" active />
              <ArchNode icon={<Cpu className="w-5 h-5 text-accent" />} label="Docking" active />
            </div>

            {/* Arrows */}
            <div className="text-text-secondary text-xs">Processing Pipeline &darr;</div>

            {/* Core Loop */}
            <div className="w-full max-w-lg">
              <div className="flex flex-col items-center gap-3 p-5 rounded-card border border-accent/30 bg-accent/5">
                <RefreshCw className="w-6 h-6 text-accent animate-spin" style={{ animationDuration: '8s' }} />
                <span className="text-sm font-semibold text-accent">
                  ASI-Evolve Agent Loop
                </span>
                <div className="grid grid-cols-3 gap-3 w-full">
                  <div className="text-center text-xs text-text-secondary bg-navy-900/60 p-2 rounded">
                    <Microscope className="w-4 h-4 mx-auto mb-1 text-warning" />
                    Critic
                  </div>
                  <div className="text-center text-xs text-text-secondary bg-navy-900/60 p-2 rounded">
                    <Brain className="w-4 h-4 mx-auto mb-1 text-accent" />
                    Strategist
                  </div>
                  <div className="text-center text-xs text-text-secondary bg-navy-900/60 p-2 rounded">
                    <FlaskConical className="w-4 h-4 mx-auto mb-1 text-success" />
                    Generator
                  </div>
                </div>
              </div>
            </div>

            {/* Arrows */}
            <div className="text-text-secondary text-xs">&darr; Output</div>

            {/* Bottom Row */}
            <div className="grid grid-cols-2 gap-4 w-full max-w-md">
              <ArchNode icon={<HardDrive className="w-5 h-5 text-success" />} label="Cognition Store" active />
              <ArchNode icon={<Layers className="w-5 h-5 text-success" />} label="Discovery Feed" active />
            </div>
          </div>
        </Section>

        {/* Data Sources */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <Section
            icon={<Database className="w-5 h-5 text-warning" />}
            title="ChEMBL"
          >
            <p>
              ChEMBL is a manually curated database of bioactive molecules with drug-like
              properties. It provides quantitative interaction data including binding affinities
              (Ki, Kd, IC50, EC50) for over 2 million compounds against thousands of targets.
            </p>
            <p>
              ASI-Evolve queries ChEMBL to retrieve structure-activity relationship (SAR) data
              for the target protein of interest, which serves as the training foundation for
              the predictive affinity model.
            </p>
          </Section>

          <Section
            icon={<Server className="w-5 h-5 text-warning" />}
            title="Protein Data Bank"
          >
            <p>
              The Protein Data Bank (PDB) provides 3D structural data for biological macromolecules.
              ASI-Evolve uses PDB structures to prepare receptor files for molecular docking simulations.
            </p>
            <p>
              Receptor preparation includes protonation, cofactor identification, and binding
              site definition to ensure accurate docking poses and reliable scoring.
            </p>
          </Section>

          <Section
            icon={<Globe className="w-5 h-5 text-warning" />}
            title="SwissADME"
          >
            <p>
              SwissADME is a web tool to predict physicochemical properties, pharmacokinetics,
              drug-likeness, and medicinal chemistry friendliness of small molecules.
            </p>
            <p>
              ASI-Evolve integrates SwissADME predictions to evaluate absorption, distribution,
              metabolism, excretion, and toxicity (ADMET) profiles for each generated candidate.
            </p>
          </Section>
        </div>

        {/* Methodology */}
        <Section
          icon={<Cpu className="w-5 h-5 text-accent" />}
          title="Methodology"
        >
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-semibold text-text-primary mb-2">
                Morgan Fingerprinting
              </h3>
              <p>
                Each molecule is encoded as a 2048-bit Morgan fingerprint (circular fingerprint,
                radius 2), which captures the local chemical environment of each atom. This
                bit-vector representation enables rapid similarity comparisons and serves as
                input to the machine learning model.
              </p>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-text-primary mb-2">
                RandomForest Affinity Model
              </h3>
              <p>
                A RandomForest regressor with 200 estimators is trained on ChEMBL bioactivity data.
                The model maps molecular fingerprints to predicted binding affinity (pKi values,
                converted to nM). Feature importance analysis guides the agent in identifying
                structural modifications most likely to improve binding.
              </p>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-text-primary mb-2">
                Molecular Docking
              </h3>
              <p>
                Generated candidates undergo rigid-receptor docking against the target protein
                structure. Docking scores (estimated binding free energy in kcal/mol) provide
                a physics-based complement to the statistical affinity prediction. A threshold
                of -7.0 kcal/mol is used as the passing criterion.
              </p>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-text-primary mb-2">
                ADMET Evaluation
              </h3>
              <p>
                Each candidate is evaluated for drug-likeness using Lipinski's Rule of Five,
                synthetic accessibility, and key pharmacokinetic properties. The overall ADMET
                score aggregates molecular weight, LogP, H-bond donors/acceptors, TPSA, and
                rotatable bond counts into a normalized drug-likeness metric.
              </p>
            </div>
          </div>
        </Section>

        {/* ASI-Evolve Loop */}
        <Section
          icon={<RefreshCw className="w-5 h-5 text-accent" />}
          title="ASI-Evolve Agent Loop"
        >
          <p className="mb-4">
            The ASI-Evolve loop is an autonomous multi-agent system that iteratively generates,
            evaluates, and learns from molecular candidates. Each cycle follows a structured
            pipeline with three specialized agent roles:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
            <div className="bg-navy-900/60 p-4 rounded border border-navy-700/50">
              <div className="flex items-center gap-2 mb-3">
                <Microscope className="w-5 h-5 text-warning" />
                <h3 className="text-sm font-semibold text-text-primary">Critic Agent</h3>
              </div>
              <p className="text-xs text-text-secondary leading-relaxed">
                Analyzes the current best candidate and identifies weaknesses. Reviews docking
                poses, ADMET failures, and suboptimal affinity predictions. Produces a detailed
                critique with specific recommendations for structural modifications.
              </p>
            </div>

            <div className="bg-navy-900/60 p-4 rounded border border-navy-700/50">
              <div className="flex items-center gap-2 mb-3">
                <Brain className="w-5 h-5 text-accent" />
                <h3 className="text-sm font-semibold text-text-primary">Strategist Agent</h3>
              </div>
              <p className="text-xs text-text-secondary leading-relaxed">
                Synthesizes the critic's feedback into a concrete modification strategy. Uses
                domain knowledge of medicinal chemistry and the RandomForest feature importance
                to prioritize which functional groups to add, remove, or modify.
              </p>
            </div>

            <div className="bg-navy-900/60 p-4 rounded border border-navy-700/50">
              <div className="flex items-center gap-2 mb-3">
                <FlaskConical className="w-5 h-5 text-success" />
                <h3 className="text-sm font-semibold text-text-primary">Generator Agent</h3>
              </div>
              <p className="text-xs text-text-secondary leading-relaxed">
                Applies the strategist's plan to propose a new SMILES string. Validates the
                proposed molecule for chemical validity, synthesizability, and novelty before
                submission to the evaluation pipeline.
              </p>
            </div>
          </div>

          <p className="mt-4">
            After each cycle, results are stored in the Cognition Store &mdash; a persistent
            memory of lessons learned, successful modifications, and failure patterns. This
            accumulated knowledge guides future cycles, enabling the system to improve its
            hit rate over time without human intervention.
          </p>
        </Section>

        {/* Cognition Store */}
        <Section
          icon={<HardDrive className="w-5 h-5 text-success" />}
          title="Cognition Store"
        >
          <p>
            The Cognition Store is the system's long-term memory. It persists learnings across
            sessions and targets, building a growing knowledge base of structure-activity
            relationships. Key capabilities include:
          </p>
          <ul className="list-disc list-inside space-y-2 mt-3">
            <li>
              <strong className="text-text-primary">Lesson Memory:</strong> Each cycle's
              critique, strategy, and outcome is stored for future reference.
            </li>
            <li>
              <strong className="text-text-primary">Pattern Recognition:</strong> Recurring
              motifs in successful modifications are identified and prioritized.
            </li>
            <li>
              <strong className="text-text-primary">Cross-Target Transfer:</strong> Knowledge
              gained on one target can inform strategies for structurally similar targets.
            </li>
            <li>
              <strong className="text-text-primary">Evidence Generation:</strong> Each
              successful discovery is accompanied by an auto-generated PDF report with full
              methodology, scores, and supporting data.
            </li>
          </ul>
        </Section>

        {/* Tech Stack */}
        <Section
          icon={<Layers className="w-5 h-5 text-accent" />}
          title="Technology Stack"
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
            {[
              { label: 'React 18', desc: 'UI Framework' },
              { label: 'TypeScript', desc: 'Type Safety' },
              { label: 'Tailwind CSS', desc: 'Styling' },
              { label: 'Vite', desc: 'Build Tool' },
              { label: 'FastAPI', desc: 'Backend API' },
              { label: 'RDKit', desc: 'Cheminformatics' },
              { label: 'Scikit-learn', desc: 'ML Models' },
              { label: 'AutoDock Vina', desc: 'Docking' },
            ].map((tech) => (
              <div
                key={tech.label}
                className="bg-navy-900/60 p-3 rounded border border-navy-700/50 text-center"
              >
                <p className="text-sm font-medium text-text-primary">{tech.label}</p>
                <p className="text-xs text-text-secondary mt-1">{tech.desc}</p>
              </div>
            ))}
          </div>
        </Section>
      </main>

      {/* Footer */}
      <footer className="border-t border-navy-700/50 mt-12">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-center text-xs text-text-secondary">
          ASI-Evolve Discovery Engine &copy; {new Date().getFullYear()}
        </div>
      </footer>
    </div>
  );
}
