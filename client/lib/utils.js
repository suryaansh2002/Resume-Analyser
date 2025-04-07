export function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60

  return {
    minutes,
    seconds: remainingSeconds,
  }
}

export function calculateEstimatedTime(numResumes) {
  // Assuming 45 seconds per resume
  const totalSeconds = numResumes * 45
  return formatTime(totalSeconds)
}

