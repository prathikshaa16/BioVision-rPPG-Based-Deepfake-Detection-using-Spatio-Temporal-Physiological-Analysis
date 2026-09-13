import React, { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { PAGE_TITLES } from './lib/titles'
import DefaultLayout from './layouts/DefaultLayout'
import Home from './pages/Home'
import Login from './pages/Login'
import Register from './pages/Register'
import Dashboard from './pages/Dashboard'
import ModelInfo from './pages/ModelInfo'
import Analysis from './pages/Analysis'
import Results from './pages/Results'
import Metrics from './pages/Metrics'
import History from './pages/History'
import Learn from './pages/Learn'
import About from './pages/About'
import Contact from './pages/Contact'
import Settings from './pages/Settings'

function RouteTitle() {
  const { pathname } = useLocation()
  useEffect(() => {
    document.title = PAGE_TITLES[pathname] || 'BioVision — See Beyond the Frame'
  }, [pathname])
  return null
}

export default function App() {
  return (
    <BrowserRouter>
      <RouteTitle />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route element={<DefaultLayout />}>
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/results" element={<Results />} />
          <Route path="/architecture" element={<ModelInfo />} />
          <Route path="/model" element={<ModelInfo />} />
          <Route path="/evaluation" element={<Metrics />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/how-it-works" element={<Learn />} />
          <Route path="/learn" element={<Learn />} />
          <Route path="/history" element={<History />} />
          <Route path="/about" element={<About />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
