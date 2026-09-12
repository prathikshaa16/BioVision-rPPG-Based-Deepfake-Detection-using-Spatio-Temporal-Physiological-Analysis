import React, { useState } from 'react'
import Card from '../components/Card'
import { APP_NAME } from '../lib/nav'
import { FaEnvelope, FaPaperPlane, FaQuestionCircle, FaShieldAlt } from 'react-icons/fa'

export default function Contact() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [subject, setSubject] = useState('')
  const [message, setMessage] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const body = encodeURIComponent(`Name: ${name}\nEmail: ${email}\n\n${message}`)
    const subj = encodeURIComponent(subject || `Message for ${APP_NAME}`)
    window.location.href = `mailto:hello@biovision.ai?subject=${subj}&body=${body}`
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <header className="page-head">
        <div>
          <h2 className="page-title">Contact</h2>
          <p className="page-sub">Questions, feedback, or reports about {APP_NAME}</p>
        </div>
        <span className="chip chip--info">
          <span className="status-dot bg-cyan-400" />
          hello@biovision.ai
        </span>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card title="Reach Us">
          <div className="space-y-4 text-sm text-slate-400">
            <p className="flex items-start gap-2.5">
              <FaEnvelope className="w-4 h-4 text-cyan-300 mt-0.5 flex-shrink-0" />
              <a href="mailto:hello@biovision.ai" className="text-cyan-300 hover:underline">hello@biovision.ai</a>
            </p>
            <p className="flex items-start gap-2.5">
              <FaQuestionCircle className="w-4 h-4 text-cyan-300 mt-0.5 flex-shrink-0" />
              If you suspect a deepfake and need guidance, include context about how the media was obtained so we can
              help route your inquiry appropriately.
            </p>
            <p className="flex items-start gap-2.5">
              <FaShieldAlt className="w-4 h-4 text-cyan-300 mt-0.5 flex-shrink-0" />
              For legal, privacy, or media integrity concerns, contact us directly via email.
            </p>
          </div>
        </Card>

        <div className="lg:col-span-2">
          <Card title="Send a Message" subtitle="This form opens your email client — no backend messaging service is used.">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1.5">Name</label>
                  <input
                    className="input-dark"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Your name"
                    required
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1.5">Email</label>
                  <input
                    className="input-dark"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    required
                  />
                </div>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">Subject</label>
                <input
                  className="input-dark"
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  placeholder="How can we help?"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">Message</label>
                <textarea
                  className="input-dark min-h-[140px] resize-y"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Write your message…"
                  required
                />
              </div>
              <button type="submit" className="btn btn-primary w-full sm:w-auto">
                <FaPaperPlane className="w-4 h-4" />
                Send via Email
              </button>
            </form>
          </Card>
        </div>
      </div>
    </div>
  )
}
