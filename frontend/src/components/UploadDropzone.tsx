import React, { useCallback, useEffect, useRef, useState } from 'react'
import { formatFileSize } from '../lib/format'
import { FaCloudUploadAlt, FaVideo, FaCheckCircle, FaExchangeAlt, FaTrashAlt } from 'react-icons/fa'

interface DropzoneProps {
  onFiles: (files: FileList) => void
  onClear?: () => void
  disabled?: boolean
  progress?: number
  selectedFile?: File
}

export default function UploadDropzone({ onFiles, onClear, disabled, progress, selectedFile }: DropzoneProps) {
  const [dragOver, setDragOver] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(selectedFile)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [selectedFile])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      if (e.dataTransfer.files && e.dataTransfer.files.length) onFiles(e.dataTransfer.files)
    },
    [onFiles]
  )

  const isUploading = disabled || typeof progress === 'number'

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => {
          if (!disabled && !selectedFile) fileInputRef.current?.click()
        }}
        className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-200 overflow-hidden ${
          dragOver
            ? 'border-cyan-400 bg-cyan-400/5 scale-[1.01] shadow-[0_0_40px_rgba(34,211,238,0.15)]'
            : 'border-slate-700 bg-slate-900/30 hover:border-slate-500'
        } ${isUploading ? 'opacity-60 pointer-events-none' : 'cursor-pointer'}`}
      >
        {!selectedFile ? (
          <>
            <div className="flex items-center justify-center mb-4">
              <div className="h-20 w-20 rounded-2xl bg-gradient-to-br from-cyan-500/15 to-blue-600/15 border border-cyan-400/20 flex items-center justify-center pulse-glow">
                <FaCloudUploadAlt className="w-9 h-9 text-cyan-300" />
              </div>
            </div>
            <p className="text-slate-100 font-semibold text-lg">Drag &amp; drop your video here</p>
            <p className="text-slate-500 text-sm mt-1">or click to browse from your computer</p>
            <div className="mt-5">
              <button
                type="button"
                disabled={disabled}
                onClick={(e) => {
                  e.stopPropagation()
                  fileInputRef.current?.click()
                }}
                className="btn btn-primary disabled:opacity-40"
              >
                <FaCloudUploadAlt className="w-4 h-4" />
                Browse Video
              </button>
              <input
                ref={fileInputRef}
                disabled={disabled}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files.length) onFiles(e.target.files)
                  e.target.value = ''
                }}
              />
            </div>
            <div className="mt-5 inline-flex flex-wrap items-center justify-center gap-2 text-[11px] text-slate-500">
              <span className="px-2.5 py-1 rounded-full bg-slate-800/70 border border-slate-700/70">MP4</span>
              <span className="px-2.5 py-1 rounded-full bg-slate-800/70 border border-slate-700/70">MOV</span>
              <span className="px-2.5 py-1 rounded-full bg-slate-800/70 border border-slate-700/70">MKV</span>
              <span className="px-2.5 py-1 rounded-full bg-slate-800/70 border border-slate-700/70">AVI</span>
              <span className="px-2.5 py-1 rounded-full bg-slate-800/70 border border-slate-700/70">WebM</span>
              <span className="px-2.5 py-1 rounded-full bg-slate-800/70 border border-slate-700/70">max 500 MB</span>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center">
            {previewUrl && selectedFile.type.startsWith('video') && (
              <video
                src={previewUrl}
                controls
                muted
                playsInline
                preload="metadata"
                className="max-h-56 rounded-xl bg-black mb-4 w-full object-contain border border-slate-700/70"
              />
            )}
            <div className="flex items-center gap-3 mb-2">
              <div className="h-10 w-10 rounded-xl bg-emerald-500/10 border border-emerald-400/30 flex items-center justify-center text-emerald-400">
                <FaCheckCircle className="w-5 h-5" />
              </div>
              <div className="text-left">
                <p className="text-slate-100 font-semibold max-w-xs truncate flex items-center gap-2">
                  <FaVideo className="w-4 h-4 text-cyan-400" />
                  {selectedFile.name}
                </p>
                <p className="text-slate-500 text-xs">{formatFileSize(selectedFile.size)}</p>
              </div>
            </div>
            <div className="mt-3 flex flex-wrap justify-center gap-3">
              <button
                type="button"
                disabled={disabled}
                onClick={(e) => {
                  e.stopPropagation()
                  fileInputRef.current?.click()
                }}
                className="btn btn-outline text-sm px-4 py-2"
              >
                <FaExchangeAlt className="w-3.5 h-3.5" />
                Change
              </button>
              {onClear && (
                <button
                  type="button"
                  disabled={disabled}
                  onClick={(e) => {
                    e.stopPropagation()
                    onClear()
                  }}
                  className="btn btn-outline text-sm px-4 py-2"
                >
                  <FaTrashAlt className="w-3.5 h-3.5" />
                  Remove
                </button>
              )}
              <input
                ref={fileInputRef}
                disabled={disabled}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files.length) onFiles(e.target.files)
                  e.target.value = ''
                }}
              />
            </div>
          </div>
        )}
      </div>

      {typeof progress === 'number' && progress > 0 && (
        <div className="space-y-2 glass-inset p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm font-medium text-slate-300">Uploading video…</div>
            <div className="text-sm font-semibold text-cyan-300">{Math.round(Math.min(100, Math.max(0, progress)))}%</div>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
            <div
              className="h-2 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all duration-300 shadow-[0_0_10px_rgba(34,211,238,0.5)]"
              style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
