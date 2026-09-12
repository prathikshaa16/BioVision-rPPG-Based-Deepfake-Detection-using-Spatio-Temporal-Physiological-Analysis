import React, { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  FaBars,
  FaTimes,
  FaArrowRight,
  FaFilm,
  FaFingerprint,
  FaChartLine,
  FaShieldAlt,
  FaRobot,
  FaDatabase,
  FaEye,
  FaVideo,
  FaBookOpen,
  FaUsers,
  FaMicrochip,
  FaLayerGroup,
  FaHourglassHalf,
  FaFileAlt,
  FaCheckCircle,
} from 'react-icons/fa'
import { APP_NAME, APP_TAGLINE, landingNav } from '../lib/nav'
import { isBackendOnline } from '../lib/api'

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i = 0) => ({ opacity: 1, y: 0, transition: { delay: i * 0.08, duration: 0.5, ease: 'easeOut' as const } }),
}

function Navbar() {
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()
  return (
    <header className="sticky top-0 z-50 border-b border-slate-800/60 bg-[#05080f]/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-5 lg:px-8 h-16 flex items-center justify-between gap-4">
        <Link to="/" className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-cyan-400 via-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold text-sm shadow-[0_0_20px_rgba(34,211,238,0.4)]">
            BV
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold text-slate-100">{APP_NAME}</div>
            <div className="text-[11px] text-slate-500">Deepfake Detection &amp; Video Forensics</div>
          </div>
        </Link>

        <nav className="hidden lg:flex items-center gap-7">
          {landingNav.map((link) =>
            link.to ? (
              <Link key={link.label} to={link.to} className="text-sm text-slate-300 hover:text-cyan-300 transition-colors">
                {link.label}
              </Link>
            ) : (
              <a key={link.label} href={link.href} className="text-sm text-slate-300 hover:text-cyan-300 transition-colors">
                {link.label}
              </a>
            )
          )}
        </nav>

        <div className="hidden lg:flex items-center gap-3">
          <Link to="/login" className="text-sm text-slate-300 hover:text-white transition-colors">
            Sign In
          </Link>
          <button onClick={() => navigate('/analysis')} className="btn btn-primary py-2 px-4 text-sm">
            Get Started
          </button>
        </div>

        <button
          className="lg:hidden p-2 rounded-lg bg-slate-800/70 text-slate-300"
          onClick={() => setOpen((v) => !v)}
          aria-label="Toggle menu"
        >
          {open ? <FaTimes className="w-5 h-5" /> : <FaBars className="w-5 h-5" />}
        </button>
      </div>

      {open && (
        <nav className="lg:hidden border-t border-slate-800/60 bg-[#05080f]/95 px-5 py-4 space-y-1">
          {landingNav.map((link) =>
            link.to ? (
              <Link
                key={link.label}
                to={link.to}
                onClick={() => setOpen(false)}
                className="block px-3 py-2 rounded-lg text-sm text-slate-300 hover:bg-slate-800/60"
              >
                {link.label}
              </Link>
            ) : (
              <a
                key={link.label}
                href={link.href}
                onClick={() => setOpen(false)}
                className="block px-3 py-2 rounded-lg text-sm text-slate-300 hover:bg-slate-800/60"
              >
                {link.label}
              </a>
            )
          )}
          <div className="pt-2 flex gap-3">
            <Link to="/login" className="btn btn-outline flex-1 text-sm py-2">
              Sign In
            </Link>
            <button onClick={() => navigate('/analysis')} className="btn btn-primary flex-1 text-sm py-2">
              Get Started
            </button>
          </div>
        </nav>
      )}
    </header>
  )
}

function ForensicVisual() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.7, ease: 'easeOut' }}
      className="relative"
    >
      <div className="absolute -inset-8 rounded-full bg-cyan-500/10 blur-3xl" aria-hidden />
      <div className="glass-card--accent p-5 relative">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-rose-500/80" />
            <span className="w-3 h-3 rounded-full bg-amber-400/80" />
            <span className="w-3 h-3 rounded-full bg-emerald-400/80" />
          </div>
          <span className="text-[11px] font-medium text-slate-400 tracking-wide">Forensic viewport · UI illustration</span>
        </div>

        <div className="relative rounded-xl overflow-hidden bg-[#04070e] border border-slate-800 h-52 md:h-60">
          <div className="absolute inset-0 opacity-40" style={{ backgroundImage: 'linear-gradient(rgba(34,211,238,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.06) 1px, transparent 1px)', backgroundSize: '26px 26px' }} />
          <span className="scan-line" />
          <div className="absolute inset-4 border border-cyan-400/40 rounded-lg">
            <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-cyan-300 rounded-tl" />
            <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-cyan-300 rounded-tr" />
            <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-cyan-300 rounded-bl" />
            <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-cyan-300 rounded-br" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center">
              <div className="w-28 h-32 md:w-32 md:h-36 border-2 border-dashed border-cyan-300/70 rounded-xl flex items-center justify-center bg-cyan-400/5">
                <span className="text-[10px] font-semibold text-cyan-300 tracking-widest uppercase px-2 py-1 rounded bg-black/60 border border-cyan-400/30">
                  Face region
                </span>
              </div>
            </div>
          </div>
          <div className="absolute bottom-3 left-4 text-[10px] text-slate-500 font-mono">
            frame_04 / 15 · sampled
          </div>
        </div>

        <div className="mt-4 grid grid-cols-8 gap-1.5">
          {Array.from({ length: 15 }).map((_, i) => (
            <div
              key={i}
              className={`aspect-video rounded-md border ${i === 3 ? 'border-cyan-400 bg-cyan-400/15 shadow-[0_0_12px_rgba(34,211,238,0.4)]' : 'border-slate-700 bg-slate-800/50'}`}
            />
          ))}
        </div>
        <div className="text-[10px] text-slate-500 mt-2 text-center">15 uniform samples → MTCNN → EfficientNet-B4</div>

        <div className="mt-4 grid grid-cols-3 gap-3 text-center">
          <div className="glass-inset p-3">
            <FaMicrochip className="w-4 h-4 text-cyan-400 mx-auto mb-1.5" />
            <div className="text-[11px] text-slate-300 font-medium">EfficientNet-B4</div>
            <div className="text-[10px] text-slate-500">per-face inference</div>
          </div>
          <div className="glass-inset p-3">
            <FaFingerprint className="w-4 h-4 text-blue-400 mx-auto mb-1.5" />
            <div className="text-[11px] text-slate-300 font-medium">MTCNN</div>
            <div className="text-[10px] text-slate-500">face detection</div>
          </div>
          <div className="glass-inset p-3">
            <FaChartLine className="w-4 h-4 text-emerald-400 mx-auto mb-1.5" />
            <div className="text-[11px] text-slate-300 font-medium">Aggregation</div>
            <div className="text-[10px] text-slate-500">mean · median · std</div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function ProductPreview() {
  const rows = [
    { name: 'speaker_interview.mp4', verdict: 'REAL', conf: '87%', color: '#34d399' },
    { name: 'press_briefing_0421.mp4', verdict: 'FAKE', conf: '93%', color: '#fb7185' },
    { name: 'social_clip_short.mp4', verdict: 'UNCERTAIN', conf: '51%', color: '#fbbf24' },
    { name: 'archived_footage_2019.mp4', verdict: 'REAL', conf: '79%', color: '#34d399' },
  ]
  return (
    <motion.div
      initial={{ opacity: 0, y: 32 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.7, ease: 'easeOut' }}
      className="relative max-w-5xl mx-auto"
    >
      <div className="absolute -inset-10 rounded-full bg-blue-600/10 blur-[120px]" aria-hidden />
      <div className="glass-card--accent p-4 md:p-6 relative">
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-2.5">
            <span className="h-8 w-8 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-white font-bold text-xs">
              BV
            </span>
            <div className="leading-tight">
              <div className="text-xs font-semibold text-slate-100">BioVision</div>
              <div className="text-[10px] text-slate-500">Dashboard</div>
            </div>
          </div>
          <span className="text-[10px] text-slate-500">Product preview · illustrative</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[190px_1fr] gap-4">
          <div className="hidden md:flex flex-col gap-1.5">
            {['Dashboard', 'Analysis', 'Results', 'History', 'Metrics', 'Model'].map((label, i) => (
              <div
                key={label}
                className={`text-xs px-3 py-2 rounded-lg border ${i === 0 ? 'bg-cyan-500/15 text-cyan-300 border-cyan-400/25' : 'text-slate-500 border-transparent'}`}
              >
                {label}
              </div>
            ))}
          </div>
          <div className="space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { v: '12', l: 'Total Analyses', c: 'text-cyan-300' },
                { v: '7', l: 'Real', c: 'text-emerald-400' },
                { v: '3', l: 'Fake', c: 'text-rose-400' },
                { v: '82%', l: 'Avg Confidence', c: 'text-blue-400' },
              ].map((s) => (
                <div key={s.l} className="glass-inset p-3">
                  <div className={`text-xl font-bold ${s.c}`}>{s.v}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5">{s.l}</div>
                </div>
              ))}
            </div>

            <div className="glass-inset overflow-hidden">
              <div className="px-4 py-2.5 text-[11px] font-semibold text-slate-400 border-b border-slate-800">
                Recent Activity
              </div>
              <div className="divide-y divide-slate-800/60">
                {rows.map((r) => (
                  <div key={r.name} className="flex items-center justify-between px-4 py-2.5">
                    <div className="flex items-center gap-2.5 min-w-0">
                      <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: r.color }} />
                      <span className="text-xs text-slate-300 truncate">{r.name}</span>
                    </div>
                    <span
                      className="text-[10px] font-semibold px-2 py-0.5 rounded-full border flex-shrink-0"
                      style={{ color: r.color, borderColor: `${r.color}55`, background: `${r.color}14` }}
                    >
                      {r.verdict} · {r.conf}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function SectionHeading({ eyebrow, title, sub }: { eyebrow: string; title: React.ReactNode; sub?: string }) {
  return (
    <div className="text-center max-w-2xl mx-auto mb-12">
      <div className="eyebrow justify-center mb-3">{eyebrow}</div>
      <h2 className="text-3xl md:text-4xl font-bold text-slate-50 tracking-tight">{title}</h2>
      {sub && <p className="text-slate-400 mt-3 text-base leading-relaxed">{sub}</p>}
    </div>
  )
}

const CAPABILITIES = [
  { icon: FaFilm, value: '15 frames', label: 'Uniformly sampled per video for representative coverage' },
  { icon: FaFingerprint, value: 'Face-level', label: 'MTCNN isolates faces to analyze identity regions' },
  { icon: FaChartLine, value: 'Per-frame', label: 'Sigmoid probability produced for every sampled frame' },
  { icon: FaShieldAlt, value: '3 verdicts', label: 'REAL · FAKE · UNCERTAIN with transparent thresholds' },
]

const HOW_IT_WORKS = [
  { icon: FaVideo, step: '01', title: 'Upload video', desc: 'Drop an MP4, MOV, MKV, AVI, or WebM file — up to 500 MB.' },
  { icon: FaFilm, step: '02', title: 'Extract frames', desc: 'The pipeline samples 15 frames at uniform intervals.' },
  { icon: FaFingerprint, step: '03', title: 'Detect faces', desc: 'MTCNN locates faces and crops the largest region per frame.' },
  { icon: FaRobot, step: '04', title: 'EfficientNet-B4 analysis', desc: 'Each face crop runs through the forensics model.' },
  { icon: FaLayerGroup, step: '05', title: 'Aggregate predictions', desc: 'Mean, median, and std combine the per-frame signals.' },
  { icon: FaFileAlt, step: '06', title: 'Generate forensic report', desc: 'A transparent verdict, confidence, and frame chart.' },
]

const FEATURES = [
  { icon: FaFingerprint, title: 'Face-level analysis', desc: 'Detection runs on detected face regions, not raw pixels, matching how manipulation is introduced.' },
  { icon: FaChartLine, title: 'Frame-level probabilities', desc: 'Every sampled frame yields an explicit fake probability you can inspect in the report.' },
  { icon: FaShieldAlt, title: 'Confidence scoring', desc: 'A normalized confidence expresses how far predictions sit from the decision boundary.' },
  { icon: FaHourglassHalf, title: 'Temporal consistency', desc: 'Cross-frame std deviation surfaces whether evidence is stable or frame-to-frame disagreement exists.' },
  { icon: FaDatabase, title: 'Detection history', desc: 'Every real result is stored locally and synchronized with the server when it is online.' },
  { icon: FaEye, title: 'Forensic explanation', desc: 'Plain-language reasoning generated from the actual per-frame model output.' },
]

const TOPICS = [
  { icon: FaBookOpen, title: 'How Deepfakes Work', desc: 'GANs, autoencoders, and diffusion models that synthesize and swap faces.' },
  { icon: FaUsers, title: 'Face Manipulation', desc: 'How identity is replaced or reenacted and where artifacts tend to appear.' },
  { icon: FaFilm, title: 'Frame-Level Detection', desc: 'Why sampling frames and analyzing faces per-frame is a robust strategy.' },
  { icon: FaChartLine, title: 'Understanding Confidence', desc: 'What a confidence score means and how to read it responsibly.' },
  { icon: FaRobot, title: 'How AI Detects Manipulation', desc: 'How EfficientNet-B4 learns to flag forensic inconsistencies.' },
]

const PLANS = [
  {
    name: 'Free',
    price: '$0',
    period: 'forever',
    tag: 'Start here',
    features: ['10 analyses / month', 'Standard video formats', 'Local detection history', 'Community support'],
    cta: 'Get Started',
    featured: false,
  },
  {
    name: 'Pro',
    price: '$19',
    period: '/ month',
    tag: 'Most popular',
    features: ['Unlimited analyses', 'Priority inference queue', 'Full forensic reports & exports', 'Frame-level visualizations', 'Priority support'],
    cta: 'Start Pro',
    featured: true,
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    period: 'tailored',
    tag: 'For teams',
    features: ['Dedicated infrastructure', 'Batch video pipelines', 'Custom model integrations', 'SSO & audit logging', 'SLAs & onboarding'],
    cta: 'Contact Sales',
    featured: false,
  },
]

function Footer() {
  return (
    <footer className="border-t border-slate-800/70 bg-[#04070e]">
      <div className="max-w-7xl mx-auto px-5 lg:px-8 py-12 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8">
        <div className="col-span-2">
          <div className="flex items-center gap-3 mb-3">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-white font-bold text-sm">
              BV
            </div>
            <div>
              <div className="text-sm font-semibold text-slate-100">{APP_NAME}</div>
              <div className="text-[11px] text-slate-500">{APP_TAGLINE}</div>
            </div>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed max-w-xs">
            AI-powered deepfake detection and video forensics built on EfficientNet-B4 and the MTCNN face detector.
          </p>
        </div>
        <div>
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Product</div>
          <ul className="space-y-2 text-sm text-slate-500">
            <li><Link className="hover:text-cyan-300 transition-colors" to="/analysis">Analysis</Link></li>
            <li><Link className="hover:text-cyan-300 transition-colors" to="/dashboard">Dashboard</Link></li>
            <li><Link className="hover:text-cyan-300 transition-colors" to="/metrics">Metrics</Link></li>
            <li><Link className="hover:text-cyan-300 transition-colors" to="/model">Model</Link></li>
          </ul>
        </div>
        <div>
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Learning</div>
          <ul className="space-y-2 text-sm text-slate-500">
            <li><Link className="hover:text-cyan-300 transition-colors" to="/learn">Learn Deepfakes</Link></li>
            <li><a className="hover:text-cyan-300 transition-colors" href="#how-it-works">How It Works</a></li>
            <li><a className="hover:text-cyan-300 transition-colors" href="#pricing">Pricing</a></li>
            <li><Link className="hover:text-cyan-300 transition-colors" to="/about">About</Link></li>
          </ul>
        </div>
        <div>
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Company</div>
          <ul className="space-y-2 text-sm text-slate-500">
            <li><Link className="hover:text-cyan-300 transition-colors" to="/about">About Us</Link></li>
            <li><Link className="hover:text-cyan-300 transition-colors" to="/contact">Contact</Link></li>
            <li><Link className="hover:text-cyan-300 transition-colors" to="/settings">Settings</Link></li>
            <li><Link className="hover:text-cyan-300 transition-colors" to="/login">Sign In</Link></li>
          </ul>
        </div>
        <div>
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Legal</div>
          <ul className="space-y-2 text-sm text-slate-500">
            <li><span className="cursor-default">Privacy</span></li>
            <li><span className="cursor-default">Terms</span></li>
            <li><span className="cursor-default">Responsible Use</span></li>
          </ul>
        </div>
      </div>
      <div className="border-t border-slate-800/60 py-5 text-center text-xs text-slate-600">
        © {new Date().getFullYear()} {APP_NAME}. Deepfake detection is probabilistic — verify before acting.
      </div>
    </footer>
  )
}

export default function Home() {
  const navigate = useNavigate()
  const [online, setOnline] = useState<boolean | null>(null)

  useEffect(() => {
    let cancelled = false
    isBackendOnline().then((ok) => {
      if (!cancelled) setOnline(ok)
    })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="min-h-screen bg-[#05080f]">
      <Navbar />

      {/* HERO */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none" aria-hidden>
          <div className="absolute top-24 left-1/4 w-72 h-72 rounded-full bg-cyan-500/10 blur-[100px]" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 rounded-full bg-blue-600/10 blur-[120px]" />
        </div>
        <div className="max-w-7xl mx-auto px-5 lg:px-8 pt-16 pb-20 lg:pt-24 lg:pb-28 grid grid-cols-1 lg:grid-cols-2 gap-14 items-center relative">
          <div>
            <motion.div variants={fadeUp} initial="hidden" animate="visible" custom={0} className="mb-5">
              <span className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-300">
                <FaShieldAlt className="w-3.5 h-3.5" />
                AI Deepfake Detection &amp; Video Forensics
              </span>
            </motion.div>
            <motion.h1
              variants={fadeUp}
              initial="hidden"
              animate="visible"
              custom={1}
              className="text-5xl md:text-6xl lg:text-7xl font-extrabold tracking-tight text-slate-50 leading-[1.05]"
            >
              See Beyond <span className="text-gradient">the Frame.</span>
            </motion.h1>
            <motion.p variants={fadeUp} initial="hidden" animate="visible" custom={2} className="mt-6 text-lg text-slate-400 max-w-xl leading-relaxed">
              AI-powered deepfake detection and video forensics designed to help you verify what is real.
            </motion.p>
            <motion.div variants={fadeUp} initial="hidden" animate="visible" custom={3} className="mt-8 flex flex-wrap gap-4">
              <button onClick={() => navigate('/analysis')} className="btn btn-primary px-7 py-3.5 text-base">
                Analyze a Video
                <FaArrowRight className="w-4 h-4" />
              </button>
              <a href="#how-it-works" className="btn btn-outline px-7 py-3.5 text-base">
                How It Works
              </a>
            </motion.div>
            <motion.div variants={fadeUp} initial="hidden" animate="visible" custom={4} className="mt-8 flex items-center gap-3 text-xs text-slate-500">
              <span className={`w-2 h-2 rounded-full ${online === null ? 'bg-slate-500' : online ? 'bg-emerald-400 pulse-glow' : 'bg-rose-500'}`} />
              {online === null ? 'Checking analysis server…' : online ? 'Analysis server online — ready for your first video' : 'Analysis server offline — start the backend on port 8000'}
            </motion.div>
          </div>
          <ForensicVisual />
        </div>
      </section>

      {/* CAPABILITIES / TRUST */}
      <section className="max-w-7xl mx-auto px-5 lg:px-8 pb-20">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {CAPABILITIES.map((c, i) => (
            <motion.div
              key={c.value}
              variants={fadeUp}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: '-60px' }}
              custom={i}
              className="glass-card p-5"
            >
              <c.icon className="w-6 h-6 text-cyan-400 mb-3" />
              <div className="text-2xl font-bold text-slate-100">{c.value}</div>
              <div className="text-xs text-slate-500 mt-1 leading-relaxed">{c.label}</div>
            </motion.div>
          ))}
        </div>
        <p className="text-center text-[11px] text-slate-600 mt-4">
          Product capabilities — these are pipeline characteristics, not benchmark accuracy claims.
        </p>
      </section>

      {/* HOW IT WORKS */}
      <section id="how-it-works" className="border-y border-slate-800/60 bg-[#060b15]/60 py-20">
        <div className="max-w-7xl mx-auto px-5 lg:px-8">
          <SectionHeading
            eyebrow="How It Works"
            title={<>From raw video to <span className="text-gradient">forensic verdict</span></>}
            sub="A transparent six-step pipeline that explains how BioVision reaches its conclusion."
          />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {HOW_IT_WORKS.map((s, i) => (
              <motion.div
                key={s.step}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: '-40px' }}
                custom={i}
                className="glass-card p-6 relative overflow-hidden"
              >
                <div className="absolute -top-6 -right-2 text-7xl font-extrabold text-slate-800/40 select-none">{s.step}</div>
                <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-cyan-400/20 to-blue-500/20 border border-cyan-400/30 flex items-center justify-center mb-4 relative">
                  <s.icon className="w-5 h-5 text-cyan-300" />
                </div>
                <div className="text-sm font-semibold text-slate-100 mb-1.5">{s.title}</div>
                <div className="text-sm text-slate-500 leading-relaxed">{s.desc}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section id="features" className="py-20">
        <div className="max-w-7xl mx-auto px-5 lg:px-8">
          <SectionHeading
            eyebrow="Product"
            title={<>Forensic-grade analysis, <span className="text-gradient">beautifully transparent</span></>}
            sub="Every feature works on the real output of the EfficientNet-B4 pipeline — nothing is simulated."
          />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map((f, i) => (
              <motion.div
                key={f.title}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: '-40px' }}
                custom={i}
                className="glass-card p-6"
              >
                <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-cyan-400/15 to-blue-500/15 border border-cyan-400/25 flex items-center justify-center mb-4">
                  <f.icon className="w-5 h-5 text-cyan-300" />
                </div>
                <div className="text-base font-semibold text-slate-100 mb-1.5">{f.title}</div>
                <div className="text-sm text-slate-500 leading-relaxed">{f.desc}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* LEARNING */}
      <section id="learn" className="border-y border-slate-800/60 bg-[#060b15]/60 py-20">
        <div className="max-w-7xl mx-auto px-5 lg:px-8">
          <SectionHeading
            eyebrow="Learn"
            title="Learn Deepfake Detection"
            sub="Guided lessons on how deepfakes are made and how AI uncovers them."
          />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {TOPICS.map((t, i) => (
              <motion.div
                key={t.title}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: '-40px' }}
                custom={i}
                className="glass-card p-6 group cursor-pointer"
                onClick={() => navigate('/learn')}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-cyan-400/15 to-blue-500/15 border border-cyan-400/25 flex items-center justify-center">
                    <t.icon className="w-4 h-4 text-cyan-300" />
                  </div>
                  <div className="text-sm font-semibold text-slate-100 group-hover:text-cyan-300 transition-colors">{t.title}</div>
                </div>
                <div className="text-sm text-slate-500 leading-relaxed">{t.desc}</div>
                <div className="mt-4 inline-flex items-center gap-1.5 text-xs font-medium text-cyan-400">
                  Start lesson <FaArrowRight className="w-3 h-3" />
                </div>
              </motion.div>
            ))}
          </div>
          <div className="text-center mt-8">
            <button onClick={() => navigate('/learn')} className="btn btn-outline">
              <FaBookOpen className="w-4 h-4" />
              Open Learning Center
            </button>
          </div>
        </div>
      </section>

      {/* PRODUCT PREVIEW */}
      <section id="product-preview" className="py-20">
        <div className="max-w-7xl mx-auto px-5 lg:px-8">
          <SectionHeading
            eyebrow="Workspace"
            title={<>One forensic workspace, <span className="text-gradient">every answer</span></>}
            sub="A dark, focused application shell — dashboard, analysis, forensic reports, history, metrics, and model specs in one product."
          />
          <ProductPreview />
          <div className="text-center mt-8">
            <button onClick={() => navigate('/dashboard')} className="btn btn-primary px-7 py-3">
              Explore the Workspace
              <FaArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </section>

      {/* PRICING */}
      <section id="pricing" className="py-20">
        <div className="max-w-7xl mx-auto px-5 lg:px-8">
          <SectionHeading
            eyebrow="Pricing"
            title="Simple plans, transparent value"
            sub="Start free and scale with your verification workload."
          />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {PLANS.map((p, i) => (
              <motion.div
                key={p.name}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: '-40px' }}
                custom={i}
                className={p.featured ? 'glass-card--accent p-7 flex flex-col' : 'glass-card p-7 flex flex-col'}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="text-lg font-bold text-slate-100">{p.name}</div>
                  <span className={`text-[10px] font-semibold uppercase tracking-wide px-2.5 py-1 rounded-full ${p.featured ? 'bg-cyan-400/20 text-cyan-300 border border-cyan-400/40' : 'bg-slate-800 text-slate-400 border border-slate-700'}`}>
                    {p.tag}
                  </span>
                </div>
                <div className="mb-5">
                  <span className="text-4xl font-extrabold text-slate-50">{p.price}</span>
                  <span className="text-sm text-slate-500 ml-1.5">{p.period}</span>
                </div>
                <ul className="space-y-2.5 mb-7 flex-1">
                  {p.features.map((f) => (
                    <li key={f} className="flex items-start gap-2.5 text-sm text-slate-400">
                      <FaCheckCircle className={`w-4 h-4 mt-0.5 ${p.featured ? 'text-cyan-400' : 'text-slate-600'}`} />
                      {f}
                    </li>
                  ))}
                </ul>
                <button
                  onClick={() => (p.name === 'Enterprise' ? navigate('/contact') : navigate('/analysis'))}
                  className={p.featured ? 'btn btn-primary w-full' : 'btn btn-outline w-full'}
                >
                  {p.cta}
                </button>
              </motion.div>
            ))}
          </div>
          <p className="text-center text-[11px] text-slate-600 mt-6">
            Presentation pricing for UI demonstration only — no payments are processed.
          </p>
        </div>
      </section>

      {/* CTA */}
      <section className="pb-24 px-5 lg:px-8">
        <div className="max-w-5xl mx-auto glass-card--accent p-10 md:p-14 text-center relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-transparent to-blue-600/10 pointer-events-none" />
          <h2 className="text-3xl md:text-4xl font-bold text-slate-50 tracking-tight mb-4">
            Ready to verify what's real?
          </h2>
          <p className="text-slate-400 max-w-xl mx-auto mb-8">
            Upload a video and get a real forensic analysis from the EfficientNet-B4 pipeline — right now.
          </p>
          <button onClick={() => navigate('/analysis')} className="btn btn-primary px-8 py-4 text-base">
            Analyze Your First Video
            <FaArrowRight className="w-4 h-4" />
          </button>
        </div>
      </section>

      <Footer />
    </div>
  )
}
