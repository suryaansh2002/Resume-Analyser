"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import * as XLSX from "xlsx"
import { toast } from "react-hot-toast"

export default function ResultsPage() {
  const router = useRouter()
  const [results, setResults] = useState(null)
  const [jobData, setJobData] = useState(null)

  useEffect(() => {
    // Check if evaluation results exist in session storage
    const evaluationResults = sessionStorage.getItem("evaluationResults")
    const jobDataString = sessionStorage.getItem("jobData")

    if (!evaluationResults || !jobDataString) {
      toast.error("No evaluation results found")
      router.push("/")
      return
    }

    try {
      const parsedResults = JSON.parse(evaluationResults)
      const parsedJobData = JSON.parse(jobDataString)

      // Sort results by overall score in descending order
      const sortedResults = {}
      Object.keys(parsedResults)
        .sort((a, b) => parsedResults[b].overall_score - parsedResults[a].overall_score)
        .forEach((key) => {
          sortedResults[key] = parsedResults[key]
        })

      setResults(sortedResults)
      setJobData(parsedJobData)
    } catch (error) {
      console.error("Error parsing results:", error)
      toast.error("Error loading results")
      router.push("/")
    }
  }, [router])

  const getScoreColor = (score) => {
    if (score >= 0.8) return "text-green-400"
    if (score >= 0.6) return "text-blue-400"
    if (score >= 0.4) return "text-yellow-400"
    return "text-red-400"
  }

  const getScoreBackgroundColor = (score) => {
    if (score >= 0.8) return "bg-green-900/20"
    if (score >= 0.6) return "bg-blue-900/20"
    if (score >= 0.4) return "bg-yellow-900/20"
    return "bg-red-900/20"
  }

  const downloadExcel = () => {
    if (!results) return

    // Create worksheet
    const worksheet = XLSX.utils.json_to_sheet(
      Object.entries(results).map(([name, scores]) => ({
        Name: name,
        "Technical Skills": scores.technical_skills.toFixed(2),
        "Soft Skills": scores.soft_skills.toFixed(2),
        Certifications: scores.certifications.toFixed(2),
        "Relevant Experience": scores.relevant_experience.toFixed(2),
        Education: scores.education.toFixed(2),
        Location: scores.location.toFixed(2),
        "Overall Score": scores.overall_score.toFixed(2),
      })),
    )

    // Create workbook
    const workbook = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(workbook, worksheet, "Resume Evaluation")

    // Generate Excel file and download
    XLSX.writeFile(workbook, `Resume_Evaluation_${jobData?.position || "Results"}.xlsx`)
  }

  if (!results || !jobData) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center p-4">
        <div className="card max-w-4xl w-full text-center">
          <p className="text-xl">Loading results...</p>
        </div>
      </main>
    )
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-4 sm:p-8 md:p-12">
      <div className="card max-w-5xl w-full">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">Evaluation Results</h1>
          <div className="mt-2 text-gray-300">
            <p>
              <span className="text-gray-400">Company:</span> {jobData.company}
            </p>
            <p>
              <span className="text-gray-400">Position:</span> {jobData.position}
            </p>
          </div>
        </div>

        <div className="space-y-6">
          {Object.entries(results).map(([name, scores], index) => (
            <div key={name} className="card bg-gray-800/50 border border-gray-700">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
                <div className="flex items-center">
                  <div className="bg-purple-600 text-white rounded-full w-8 h-8 flex items-center justify-center mr-3">
                    {index + 1}
                  </div>
                  <h2 className="text-xl font-semibold">{name}</h2>
                </div>
                <div className="mt-2 md:mt-0">
                  <span className="text-gray-400 mr-2">Overall Score:</span>
                  <span className={`text-xl font-bold ${getScoreColor(scores.overall_score)}`}>
                    {scores.overall_score.toFixed(2)}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className={`p-3 rounded-lg ${getScoreBackgroundColor(scores.technical_skills)}`}>
                  <p className="text-sm text-gray-400">Technical Skills</p>
                  <p className={`text-lg font-semibold ${getScoreColor(scores.technical_skills)}`}>
                    {scores.technical_skills.toFixed(2)}
                  </p>
                </div>

                <div className={`p-3 rounded-lg ${getScoreBackgroundColor(scores.soft_skills)}`}>
                  <p className="text-sm text-gray-400">Soft Skills</p>
                  <p className={`text-lg font-semibold ${getScoreColor(scores.soft_skills)}`}>
                    {scores.soft_skills.toFixed(2)}
                  </p>
                </div>

                <div className={`p-3 rounded-lg ${getScoreBackgroundColor(scores.certifications)}`}>
                  <p className="text-sm text-gray-400">Certifications</p>
                  <p className={`text-lg font-semibold ${getScoreColor(scores.certifications)}`}>
                    {scores.certifications.toFixed(2)}
                  </p>
                </div>

                <div className={`p-3 rounded-lg ${getScoreBackgroundColor(scores.relevant_experience)}`}>
                  <p className="text-sm text-gray-400">Relevant Experience</p>
                  <p className={`text-lg font-semibold ${getScoreColor(scores.relevant_experience)}`}>
                    {scores.relevant_experience.toFixed(2)}
                  </p>
                </div>

                <div className={`p-3 rounded-lg ${getScoreBackgroundColor(scores.education)}`}>
                  <p className="text-sm text-gray-400">Education</p>
                  <p className={`text-lg font-semibold ${getScoreColor(scores.education)}`}>
                    {scores.education.toFixed(2)}
                  </p>
                </div>

                <div className={`p-3 rounded-lg ${getScoreBackgroundColor(scores.location)}`}>
                  <p className="text-sm text-gray-400">Location</p>
                  <p className={`text-lg font-semibold ${getScoreColor(scores.location)}`}>
                    {scores.location.toFixed(2)}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="flex flex-col sm:flex-row justify-between mt-8">
          <button
            onClick={() => router.push("/")}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors mb-4 sm:mb-0"
          >
            Back to Home
          </button>

          <button onClick={downloadExcel} className="btn-primary flex items-center justify-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
            Download Excel
          </button>
        </div>
      </div>
    </main>
  )
}

