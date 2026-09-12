import React from 'react'
import { Link } from 'react-router-dom'
import { APP_NAME } from '../lib/nav'

export default function Register() {
  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-[#05080f]">
      <div className="w-full max-w-md">
        <div className="glass-card--accent rounded-2xl p-8">
          <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-white font-bold mb-4 shadow-[0_0_20px_rgba(34,211,238,0.4)]">
            BV
          </div>
          <h2 className="text-xl font-bold text-slate-50 mb-1">Create your {APP_NAME} account</h2>
          <p className="text-sm text-slate-500 mb-6">Start detecting deepfakes in minutes</p>
          <form className="space-y-3">
            <input className="input-dark" placeholder="Name" />
            <input className="input-dark" placeholder="Email" type="email" />
            <input className="input-dark" placeholder="Password" type="password" />
            <button className="btn btn-primary w-full py-3">Register</button>
          </form>
          <p className="mt-4 text-sm text-slate-500">
            Already have an account?{' '}
            <Link to="/login" className="text-cyan-300 hover:underline">Sign in</Link>
          </p>
          <p className="mt-6 text-[11px] text-slate-600">
            Authentication is UI-only — wire this form to your identity provider when available.
          </p>
        </div>
      </div>
    </div>
  )
}
