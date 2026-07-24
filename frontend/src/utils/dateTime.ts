const CHINA_TIME_ZONE = 'Asia/Shanghai'

type DateValue = Date | string | number | null | undefined

function parseDate(value: DateValue): Date | null {
  if (value == null || value === '') return null
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value
  if (typeof value === 'number') {
    const date = new Date(value)
    return Number.isNaN(date.getTime()) ? null : date
  }

  const normalized = value.trim().replace(' ', 'T')
  const hasTimeZone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(normalized)
  const date = new Date(hasTimeZone ? normalized : `${normalized}Z`)
  return Number.isNaN(date.getTime()) ? null : date
}

function format(value: DateValue, options: Intl.DateTimeFormatOptions, fallback = '--') {
  const date = parseDate(value)
  if (!date) return fallback
  return new Intl.DateTimeFormat('zh-CN', { timeZone: CHINA_TIME_ZONE, ...options }).format(date)
}

export function formatChinaTime(value: DateValue, includeSeconds = false) {
  return format(value, {
    hour: '2-digit',
    minute: '2-digit',
    second: includeSeconds ? '2-digit' : undefined,
    hourCycle: 'h23',
  })
}

export function formatChinaDate(value: DateValue) {
  return format(value, { year: 'numeric', month: '2-digit', day: '2-digit' })
}

export function formatChinaDateTime(value: DateValue) {
  return format(value, {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
  })
}
