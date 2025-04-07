"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import axios from "axios"
import LoadingSpinner from "@/components/loading-spinner"
import { toast } from "react-hot-toast"

export default function JobDescriptionPage() {
  const router = useRouter()
  const [jobTitle, setJobTitle] = useState("")
  const [companyName, setCompanyName] = useState("")
  const [jobDescription, setJobDescription] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState("Analyzing Job Description...")

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!jobTitle || !companyName || !jobDescription) {
      toast.error("Please fill in all fields")
      return
    }

    setIsLoading(true)

    try {
      // Format job description as requested
      const formatted_jd = `Job Title: ${jobTitle}\nCompany Name: ${companyName}\nDescription: ${jobDescription}`

      // Make API call
      const response = await axios.post("http://127.0.0.1:8000/job_description", {
        job_description: formatted_jd,
      })

      // Check if response contains position key
      if (response.data && response.data.position) {
        // Save to session storage
        sessionStorage.setItem("jobData", JSON.stringify(response.data))

        toast.success("Job description analyzed successfully!")

        // Navigate to resume page
        router.push("/resume-upload")
      } else {
        throw new Error("Invalid response from server")
      }
    } catch (error) {
      console.error("Error submitting job description:", error)
      toast.error("Failed to analyze job description. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 sm:p-24">
      <div className="card max-w-3xl w-full">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">Job Description</h1>
          <p className="text-gray-400 mt-2">Enter the details of the position you're hiring for</p>
        </div>

        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-12">
            <LoadingSpinner />
            <p className="mt-4 text-lg text-gray-300">{loadingMessage}</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="jobTitle" className="block text-sm font-medium text-gray-300 mb-1">
                Job Title
              </label>
              <input
                id="jobTitle"
                type="text"
                value={jobTitle}
                onChange={(e) => setJobTitle(e.target.value)}
                className="input-field"
                placeholder="e.g. Senior Software Engineer"
                required
              />
            </div>

            <div>
              <label htmlFor="companyName" className="block text-sm font-medium text-gray-300 mb-1">
                Company Name
              </label>
              <input
                id="companyName"
                type="text"
                value={companyName}
                onChange={(e) => setCompanyName(e.target.value)}
                className="input-field"
                placeholder="e.g. Tech Innovations Inc."
                required
              />
            </div>

            <div>
              <label htmlFor="jobDescription" className="block text-sm font-medium text-gray-300 mb-1">
                Job Description
              </label>
              <textarea
                id="jobDescription"
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                className="input-field min-h-[200px]"
                placeholder="Enter the full job description here..."
                required
              />
            </div>

            <div className="flex justify-between pt-4">
              <button
                type="button"
                onClick={() => router.push("/")}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                Back
              </button>

              <button type="submit" className="btn-primary">
                Continue
              </button>
            </div>
          </form>
        )}
      </div>
    </main>
  )
}

