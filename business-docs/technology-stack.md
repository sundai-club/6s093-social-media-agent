# Emanon - Technology Stack

## Frontend

| Technology | Purpose |
|------------|---------|
| **Astro** | Static site generation with dynamic islands |
| **Tailwind CSS v4** | Styling and design system |
| **TypeScript** | Type-safe component development |
| **MDX** | Markdown with embedded components for blog |
| **Chart.js** | Data visualization for benchmarks |

---

## Backend & APIs

| Service | Purpose |
|---------|---------|
| **Stripe** | Payment processing for subscriptions and donations |
| **Resend** | Email delivery for newsletter |
| **Anthropic Claude API** | Content translation and AI processing |
| **Calendly** | Consultation booking |
| **LinkedIn API** | Automated posting |

---

## Infrastructure

| Platform | Purpose |
|----------|---------|
| **Cloudflare Pages** | Static site hosting |
| **Cloudflare Workers** | Serverless API functions |
| **Cloudflare** | CDN and edge computing |

---

## Build Tools

- **Node.js** with npm
- **Astro CLI** for development and building
- **Gray-matter** for frontmatter parsing

---

## Key Integrations

### Stripe Integration
- Three-tier subscription model
- One-time donations
- Webhook handling for subscription events

### Email System (Resend)
- Newsletter delivery
- Subscription management
- Context tracking

### AI Integration (Anthropic)
- Claude SDK (v0.71.0+)
- Automatic content translation
- 11+ language support

### Calendly
- Consultation booking
- 1-hour session scheduling
- Automated confirmation

---

## Architecture Notes

- **Static-first** - Astro generates static pages for performance
- **Edge computing** - Cloudflare Workers handle dynamic API needs
- **No backwards compatibility** - Old features are removed when updated, not deprecated
- **SEO optimized** - Structured data, sitemaps, site-wide search
