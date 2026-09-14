import type { IconType } from 'react-icons'
import {
  FaHome,
  FaUpload,
  FaSearch,
  FaHistory,
  FaChartBar,
  FaMicrochip,
  FaBookOpen,
  FaInfoCircle,
} from 'react-icons/fa'

export interface NavItem {
  to: string
  label: string
  icon: IconType
  end?: boolean
}

export const APP_NAME = 'BioVision'
export const APP_TAGLINE = 'Spatio-Temporal & Physiological Deepfake Detection'

export const navItems: NavItem[] = [
  { to: '/', label: 'Home', icon: FaHome, end: true },
  { to: '/analysis', label: 'Analyze', icon: FaUpload },
  { to: '/how-it-works', label: 'How It Works', icon: FaBookOpen },
  { to: '/architecture', label: 'Architecture', icon: FaMicrochip },
  { to: '/results', label: 'Results', icon: FaSearch },
  { to: '/evaluation', label: 'Evaluation', icon: FaChartBar },
  { to: '/history', label: 'History', icon: FaHistory },
  { to: '/about', label: 'About', icon: FaInfoCircle },
]

export interface LandingLink {
  label: string
  to?: string
  href?: string
}

export const landingNav: LandingLink[] = [
  { label: 'Home', to: '/' },
  { label: 'Analyze', to: '/analysis' },
  { label: 'How It Works', href: '#how-it-works' },
  { label: 'Architecture', to: '/architecture' },
  { label: 'Results', to: '/results' },
  { label: 'Evaluation', to: '/evaluation' },
  { label: 'About', to: '/about' },
]
