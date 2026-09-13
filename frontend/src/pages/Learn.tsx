import React, { useState } from 'react'
import Card from '../components/Card'
import {
  FaBookOpen,
  FaUsers,
  FaFilm,
  FaChartLine,
  FaRobot,
  FaChevronDown,
  FaVideo,
  FaLightbulb,
  FaShieldAlt,
  FaSearch,
} from 'react-icons/fa'

interface Topic {
  icon: React.ComponentType<{ className?: string }>
  title: string
  desc: string
  points: string[]
}

const TOPICS: Topic[] = [
  {
    icon: FaBookOpen,
    title: 'How Deepfakes Work',
    desc: 'The generative techniques — GANs, autoencoders, and diffusion models — that create synthetic faces.',
    points: [
      'Generative Adversarial Networks pit a generator against a discriminator to produce photorealistic output.',
      'Autoencoders learn a compact encoding of a face that can be swapped or reenacted onto another person.',
      'Diffusion models progressively denoise random noise into images, enabling high-fidelity synthesis.',
      'Detection looks for the subtle artifacts these generative processes tend to leave behind.',
    ],
  },
  {
    icon: FaUsers,
    title: 'Face Manipulation',
    desc: 'How identity replacement and expression reenactment are performed — and where artifacts appear.',
    points: [
      'Identity swap replaces one face with another while keeping the source video motion.',
      'Expression reenactment copies facial movements from a source onto a target face.',
      'Artifacts often concentrate around eyes, mouth, teeth, and hair boundaries.',
      'Lighting, skin texture, and blinking inconsistencies are common forensic signals.',
    ],
  },
  {
    icon: FaFilm,
    title: 'Visual-Temporal Analysis',
    desc: 'Why a sequence of EfficientNet-B4 embeddings and an LSTM is useful for video forensics.',
    points: [
      'A sequence preserves changes in face appearance and motion across the clip.',
      'Uniform sampling provides coverage across the full temporal range of the video.',
      'The LSTM converts the 32 embedding sequence into a 256-dimensional temporal representation.',
      'BioVision combines this visual context with a 64-dimensional CHROM-rPPG representation.',
    ],
  },
  {
    icon: FaChartLine,
    title: 'Understanding Confidence',
    desc: 'What a confidence score means, how it is computed, and how to read it responsibly.',
    points: [
      'BioVision computes confidence as how far the mean probability sits from the 50% decision boundary.',
      'High confidence means the mean was far from the boundary; low confidence means borderline.',
      'Confidence expresses model certainty, not the probability the verdict is correct.',
      'Use confidence to decide whether further manual review is warranted.',
    ],
  },
  {
    icon: FaSearch,
    title: 'Detecting Visual Artifacts',
    desc: 'The forensic fingerprints generated images leave behind — and where to look for them.',
    points: [
      'Generator upscaling often blurs or distorts fine facial details like teeth and ear edges.',
      'Inconsistent lighting, specular highlights, and skin texture are common tells.',
      'Face boundaries can flicker between frames where reenactment was applied.',
      'The model learns these subtle statistical cues that are hard to see with the naked eye.',
    ],
  },
  {
    icon: FaRobot,
    title: 'How AI Forensics Works',
    desc: 'How EfficientNet-B4 learns forensic indicators and why the head architecture matters.',
    points: [
      'EfficientNet-B4 is a scalable convolutional backbone pretrained on ImageNet.',
      'The visual-temporal branch produces 256 dimensions and the rPPG branch produces 64 dimensions.',
      'The trained fusion head maps the combined 320-dimensional representation to a fake probability in [0, 1].',
      'The result page exposes the probability, confidence, temporal evidence, and physiological charts.',
    ],
  },
]

function TopicCard({ topic }: { topic: Topic }) {
  const [open, setOpen] = useState(false)
  const Icon = topic.icon
  return (
    <div className="glass-card overflow-hidden">
      <button onClick={() => setOpen((v) => !v)} className="w-full text-left p-6 flex items-start justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-cyan-400/15 to-blue-500/15 border border-cyan-400/25 flex items-center justify-center flex-shrink-0">
            <Icon className="w-5 h-5 text-cyan-300" />
          </div>
          <div>
            <div className="text-base font-semibold text-slate-100">{topic.title}</div>
            <div className="text-sm text-slate-500 mt-1 leading-relaxed">{topic.desc}</div>
          </div>
        </div>
        <FaChevronDown className={`w-4 h-4 text-slate-500 flex-shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="px-6 pb-6 pt-1">
          <ul className="space-y-2.5">
            {topic.points.map((p) => (
              <li key={p} className="flex items-start gap-2.5 text-sm text-slate-400 leading-relaxed">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 flex-shrink-0" />
                {p}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default function Learn() {
  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <header className="page-head">
        <div>
          <h2 className="page-title">Learn Deepfake Detection</h2>
          <p className="page-sub">Guided lessons on how deepfakes are made and how AI uncovers them</p>
        </div>
        <span className="chip chip--info">
          <span className="status-dot bg-cyan-400" />
          {TOPICS.length} Lessons
        </span>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass-card p-5 flex items-start gap-4">
          <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-cyan-400/15 to-blue-500/15 border border-cyan-400/25 flex items-center justify-center flex-shrink-0">
            <FaVideo className="w-5 h-5 text-cyan-300" />
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-100 mb-1">Video lessons</div>
            <p className="text-xs text-slate-500 leading-relaxed">
              Guided video walkthroughs will appear here once they are produced. No lessons are being faked today —
              every topic below is a text primer you can read now.
            </p>
          </div>
        </div>
        <div className="glass-card p-5 flex items-start gap-4">
          <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-cyan-400/15 to-blue-500/15 border border-cyan-400/25 flex items-center justify-center flex-shrink-0">
            <FaLightbulb className="w-5 h-5 text-cyan-300" />
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-100 mb-1">Why it matters</div>
            <p className="text-xs text-slate-500 leading-relaxed">
              Understanding the forensic pipeline helps you read confidence scores, spot borderline cases, and verify
              media responsibly.
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {TOPICS.map((topic) => (
          <TopicCard key={topic.title} topic={topic} />
        ))}
      </div>

      <Card title="Reading BioVision Verdicts" subtitle="The exact thresholds used by the backend classification">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="border-l-4 border-emerald-500 pl-4">
            <p className="font-semibold text-emerald-400 mb-1">REAL</p>
            <p className="text-sm text-slate-500">Mean fake probability ≤ 40% — no strong manipulation indicators.</p>
          </div>
          <div className="border-l-4 border-rose-500 pl-4">
            <p className="font-semibold text-rose-400 mb-1">FAKE</p>
            <p className="text-sm text-slate-500">Mean fake probability ≥ 60% — elevated manipulation evidence.</p>
          </div>
          <div className="border-l-4 border-amber-500 pl-4">
            <p className="font-semibold text-amber-400 mb-1">UNCERTAIN</p>
            <p className="text-sm text-slate-500">Mean between 40% and 60% — manual review recommended.</p>
          </div>
        </div>
      </Card>

      <Card title="Detection Ethics" className="border-cyan-400/20">
        <div className="flex items-start gap-3">
          <FaShieldAlt className="w-5 h-5 text-cyan-300 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-slate-500 leading-relaxed">
            Detection is probabilistic and not perfect. False positives and negatives can occur. Use this tool
            responsibly, respect privacy, and consider legal and ethical implications before sharing or acting on
            results.
          </p>
        </div>
      </Card>
    </div>
  )
}
