export const metadata = {
  title: "CT vs MRI Analyzer",
  description: "Client-side analysis demo (not medical advice)",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
