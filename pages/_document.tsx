import { Html, Head, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html className="dark" lang="en">
      <Head />
      <body className="dark:bg-ghd-dark-bg">
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
