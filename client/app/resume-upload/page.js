"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import axios from "axios"
import LoadingSpinner from "@/components/loading-spinner"
import { toast } from "react-hot-toast"

export default function ResumeUploadPage() {
  const router = useRouter()
  const [file, setFile] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState("")
  const [resumeCount, setResumeCount] = useState(0)
  const [estimatedTime, setEstimatedTime] = useState({ minutes: 0, seconds: 0 })
  const [uploadProgress, setUploadProgress] = useState(0)

  useEffect(() => {
    // Check if job data exists in session storage
    const jobData = sessionStorage.getItem("jobData")
    if (!jobData) {
      toast.error("Please complete the job description first")
      router.push("/job-description")
    }
  }, [router])

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      const selectedFile = e.target.files[0]
      if (selectedFile.type === "application/zip" || selectedFile.type === "application/x-zip-compressed") {
        setFile(selectedFile)
      } else {
        toast.error("Please upload a ZIP file")
        e.target.value = null
      }
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!file) {
      toast.error("Please upload a ZIP file containing resumes")
      return
    }

    setIsLoading(true)
    setLoadingMessage("Uploading ZIP file...")
    setUploadProgress(0)

    try {
      // Create form data for file upload
      const formData = new FormData()
      formData.append("file", file)

      // Upload ZIP file
      const uploadResponse = await axios.post("http://127.0.0.1:8000/upload_zip/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          setUploadProgress(percentCompleted)
        },
      })

      console.log("Upload response:", uploadResponse.data)

      // Calculate number of resumes and estimated time
      const resumesData = uploadResponse.data
      const numResumes = Object.keys(resumesData).length
      setResumeCount(numResumes)

      const totalSeconds = numResumes * 45
      const minutes = Math.floor(totalSeconds / 60)
      const seconds = totalSeconds % 60
      setEstimatedTime({ minutes, seconds })

      setLoadingMessage(`Processing ${numResumes} resumes... Estimated time: ${minutes}m ${seconds}s`)

      // Get job data from session storage
      const jobData = JSON.parse(sessionStorage.getItem("jobData"))

      // Make request to /resumes endpoint
      setLoadingMessage("Analyzing resumes...")
      const resumesResponse = await axios.post("http://127.0.0.1:8000/resumes", {
        job_title: jobData.position,
        resumes: resumesData,
      })

      // Store resume data in session storage
      sessionStorage.setItem("resumeData", JSON.stringify(resumesResponse.data))

      // Make request to /evaluate endpoint
      setLoadingMessage("Evaluating candidates...")
      const evaluateResponse = await axios.post("http://127.0.0.1:8000/evaluate", {
        job_data: jobData,
        resumes: resumesResponse.data,
      })

      // Store evaluation results in session storage
      sessionStorage.setItem("evaluationResults", JSON.stringify(evaluateResponse.data))

      toast.success("Resumes evaluated successfully!")

      // Navigate to results page
      router.push("/results")
    } catch (error) {
      console.error("Error processing resumes:", error)
      toast.error("Failed to process resumes. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 sm:p-24">
      <div className="card max-w-3xl w-full">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">Upload Resumes</h1>
          <p className="text-gray-400 mt-2">Upload a ZIP file containing all candidate resumes (PDF format)</p>
        </div>

        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-12">
            <LoadingSpinner />
            <p className="mt-4 text-lg text-gray-300">{loadingMessage}</p>

            {uploadProgress > 0 && uploadProgress < 100 && (
              <div className="w-full max-w-md mt-4">
                <div className="bg-gray-700 rounded-full h-2.5 mt-2">
                  <div className="bg-purple-600 h-2.5 rounded-full" style={{ width: `${uploadProgress}%` }}></div>
                </div>
                <p className="text-sm text-gray-400 mt-1 text-right">{uploadProgress}%</p>
              </div>
            )}

            {resumeCount > 0 && (
              <div className="mt-6 text-center">
                <p className="text-gray-300">Processing {resumeCount} resumes</p>
                <p className="text-gray-400 text-sm mt-1">
                  Estimated time: {estimatedTime.minutes}m {estimatedTime.seconds}s
                </p>
              </div>
            )}
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center">
              <input id="file-upload" type="file" onChange={handleFileChange} className="hidden" accept=".zip" />

              <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center justify-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-12 w-12 text-gray-400 mb-3"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>

                <span className="text-gray-300 font-medium">{file ? file.name : "Click to upload ZIP file"}</span>

                {!file && <p className="text-gray-500 text-sm mt-1">ZIP file containing PDF resumes</p>}
              </label>
            </div>

            <div className="flex justify-between pt-4">
              <button
                type="button"
                onClick={() => router.push("/job-description")}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                Back
              </button>

              <button type="submit" className="btn-primary" disabled={!file}>
                Evaluate Resumes
              </button>
            </div>
          </form>
        )}
      </div>
    </main>
  )
}

