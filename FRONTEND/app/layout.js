import "./globals.css";

export const metadata = { title: "Assistant médical SMA", description: "Assistant médical multi-agents" };
export const viewport = { width: "device-width", initialScale: 1 };

const THEME_SCRIPT = `try{var t=localStorage.getItem("sma_theme");if(t!=="light"&&t!=="dark"){t=matchMedia("(prefers-color-scheme: light)").matches?"light":"dark"}document.documentElement.dataset.theme=t}catch(e){document.documentElement.dataset.theme="dark"}`;

export default function RootLayout({ children }) {
  return (
    <html lang="fr" suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} /></head>
      <body>{children}</body>
    </html>
  );
}