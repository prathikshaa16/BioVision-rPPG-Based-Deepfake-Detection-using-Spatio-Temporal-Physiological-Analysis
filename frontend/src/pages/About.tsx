import React from 'react'
import { Link } from 'react-router-dom'
import Card from '../components/Card'
import { APP_NAME } from '../lib/nav'
import { FaEye, FaFingerprint, FaMicrochip, FaShieldAlt, FaArrowRight, FaChartLine } from 'react-icons/fa'

export default function About() {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <header className="page-head">
        <div>
          <h2 className="page-title">About {APP_NAME}</h2>
          <p className="page-sub">Spatio-temporal deep learning with physiological signal analysis</p>
        </div>
        <span className="chip chip--info">
          <span className="status-dot bg-cyan-400" />
          Trusted Detection
        </span>
      </header>

      <div className="glass-card--accent p-7">
        <div className="flex items-start gap-4">
          <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-white font-bold flex-shrink-0 shadow-[0_0_20px_rgba(34,211,238,0.4)]">
            BV
          </div>
          <div>
            <p className="text-lg font-bold text-slate-50">{APP_NAME}</p>
            <p className="text-sm text-slate-400">{APP_NAME} · AI Deepfake Detection &amp; Video Forensics</p>
          </div>
        </div>
        <p className="text-sm text-slate-400 leading-relaxed mt-5">
          {APP_NAME} is an AI deepfake detection platform that helps you verify whether a video is real. It samples
          video frames, detects faces with MTCNN, scores each face with an EfficientNet-B4 forensic classifier, and
          extracts CHROM rPPG from forehead and cheek regions. Quality-gated late fusion combines the two evidence
          channels into a transparent forensic report.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card title="Our Mission">
          <p className="text-sm text-slate-500 leading-relaxed">
            Make reliable deepfake verification accessible. We believe detection must be transparent — every verdict
            is backed by per-frame probabilities and a readable explanation.
          </p>
        </Card>
        <Card title="Our Approach">
          <p className="text-sm text-slate-500 leading-relaxed">
            Frame sampling, face-level analysis, physiological signal extraction, and quality-gated late fusion. No
            black boxes: the model, weights, thresholds, and pipeline are documented in the Model page.
          </p>
        </Card>
        <Card title="Our Commitment">
          <p className="text-sm text-slate-500 leading-relaxed">
            We never fabricate results. If the backend is offline, we say so. Benchmark claims are only reported when
            measured on public datasets.
          </p>
        </Card>
      </div>

      <Card title="How the pipeline works">
        <ul className="space-y-3">
          {[
            { icon: FaEye, t: 'Frame extraction', d: '15 uniform samples drawn across the video.' },
            { icon: FaFingerprint, t: 'Face detection', d: 'MTCNN locates faces and crops the largest region per frame.' },
            { icon: FaMicrochip, t: 'Model inference', d: 'EfficientNet-B4 scores each face crop as a fake probability.' },
            { icon: FaChartLine, t: 'Physiological analysis', d: 'CHROM extracts a temporal pulse signal and estimates its quality.' },
            { icon: FaShieldAlt, t: 'Evidence fusion', d: 'Visual and physiological evidence combine with a transparent 80/20 cap.' },
          ].map((s) => (
            <li key={s.t} className="flex gap-4">
              <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-cyan-400/15 to-blue-500/15 border border-cyan-400/25 flex items-center justify-center flex-shrink-0">
                <s.icon className="w-4 h-4 text-cyan-300" />
              </div>
              <div>
                <p className="font-medium text-slate-100 text-sm">{s.t}</p>
                <p className="text-sm text-slate-500">{s.d}</p>
              </div>
            </li>
          ))}
        </ul>
      </Card>

      <Card title="Limitations & Ethics">
        <p className="text-sm text-slate-500 leading-relaxed">
          Detection is probabilistic and not perfect. False positives and negatives can occur. Use this tool
          responsibly, respect privacy, and consider legal/ethical implications before sharing or acting on results.
          For critical decisions, always combine automated detection with manual review and additional forensic
          methods.
        </p>
      </Card>

      <div className="flex flex-col sm:flex-row gap-3">
        <Link to="/analysis" className="btn btn-primary flex-1">
          Start Analyzing
          <FaArrowRight className="w-4 h-4" />
        </Link>
        <Link to="/contact" className="btn btn-outline flex-1">
          Contact Us
        </Link>
      </div>
    </div>
  )
}
