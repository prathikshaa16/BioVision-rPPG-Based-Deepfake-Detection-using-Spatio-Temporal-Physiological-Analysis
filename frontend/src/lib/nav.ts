import type { IconType } from 'react-icons'
import {
  FaTachometerAlt,
  FaUpload,
  FaSearch,
  FaHistory,
  FaChartBar,
  FaMicrochip,
  FaBookOpen,
  FaInfoCircle,
  FaEnvelope,
  FaCog,
} from 'react-icons/fa'

export interface NavItem {
  to: string
  label: string
  icon: IconType
  end?: boolean
}

export const APP_NAME = 'BioVision'
export const APP_TAGLINE = 'AI Deepfake Detection & Video Forensics'

export const navItems: NavItem[] = [
  { to: '/dashboard', label: 'Dashboard', icon: FaTachometerAlt, end: true },
  { to: '/analysis', label: 'Analysis', icon: FaUpload },
  { to: '/results', label: 'Results', icon: FaSearch },
  { to: '/history', label: 'History', icon: FaHistory },
  { to: '/metrics', label: 'Metrics', icon: FaChartBar },
  { to: '/model', label: 'Model', icon: FaMicrochip },
  { to: '/learn', label: 'Learn', icon: FaBookOpen },
  { to: '/about', label: 'About', icon: FaInfoCircle },
  { to: '/contact', label: 'Contact', icon: FaEnvelope },
  { to: '/settings', label: 'Settings', icon: FaCog },
]

export interface LandingLink {
  label: string
  to?: string
  href?: string
}

export const landingNav: LandingLink[] = [
  { label: 'Product', href: '#features' },
  { label: 'How It Works', href: '#how-it-works' },
  { label: 'Learn', to: '/learn' },
  { label: 'Pricing', href: '#pricing' },
  { label: 'About', to: '/about' },
  { label: 'Contact', to: '/contact' },
]
