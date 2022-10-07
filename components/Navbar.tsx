import { motion } from 'framer-motion';
import Head from 'next/head';
import Link from 'next/link';

export default function NavBar() {
  return (
    <>
      <Head>
        <title>
          GHD Snippets: A very useful and professional and well designed snippet
          codex made by professionals for professionals!
        </title>
        <link rel="shortcut icon" href="/static/images/favicon.ico" />
        <meta
          name="description"
          content="GHD Snippets: A very useful and professional and well designed snippet
          codex made by professionals for professionals!"
          key="desc"
        />
      </Head>
      {/* Navbar */}
      <div className="flex h-22 md:h-28 w-full items-center justify-center">
        <div className="flex h-full w-full max-w-4xl items-center justify-between p-5 text-2xl dark:text-ghd-dark-text">
          {/* Github svg */}
          <motion.div
            drag
            dragSnapToOrigin
            dragTransition={{ bounceStiffness: 300, bounceDamping: 15 }}
            className="cursor-pointer"
          >
            <Link href="https://github.com/Gholamrezadar/ghd-snippets-next">
              <svg
                viewBox="0 0 38 38"
                className="h-9 w-9"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M19 3.16667C10.2555 3.16667 3.16667 10.2555 3.16667 19C3.16667 27.7445 10.2555 34.8333 19 34.8333C27.7445 34.8333 34.8333 27.7445 34.8333 19C34.8333 10.2555 27.7445 3.16667 19 3.16667ZM0 19C0 8.50662 8.50662 0 19 0C29.4934 0 38 8.50662 38 19C38 29.4934 29.4934 38 19 38C8.50662 38 0 29.4934 0 19Z"
                  fill="currentColor"
                />
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M15.1868 35.9983C15.0336 35.8007 15.0336 34.0395 15.1868 30.7151C13.5465 30.7718 12.5102 30.6579 12.0779 30.3736C11.4293 29.9472 10.78 28.632 10.2039 27.7051C9.62777 26.7782 8.34915 26.6317 7.83279 26.4245C7.31644 26.2172 7.18682 25.3726 9.25561 26.0114C11.3243 26.6502 11.4253 28.3897 12.0779 28.7965C12.7305 29.2032 14.2908 29.0253 14.998 28.6994C15.7052 28.3736 15.6531 27.1603 15.7793 26.6797C15.9387 26.2311 15.3768 26.1316 15.3644 26.1279C14.6739 26.1279 11.0468 25.3391 10.0506 21.8267C9.05429 18.3144 10.3377 16.0186 11.0223 15.0318C11.4788 14.3738 11.4384 12.9716 10.9012 10.825C12.8515 10.5758 14.3566 11.1896 15.4164 12.6667C15.4175 12.6752 16.8059 11.8411 19.0002 11.8411C21.1945 11.8411 21.9729 12.5205 22.5737 12.6667C23.1745 12.813 23.655 10.0811 27.3654 10.825C26.5907 12.3475 25.942 14.2501 26.4372 15.0318C26.9323 15.8134 28.8756 18.2991 27.6819 21.8267C26.8861 24.1785 25.3217 25.6122 22.9887 26.1279C22.7212 26.2133 22.5874 26.351 22.5874 26.541C22.5874 26.8261 22.9491 26.8572 23.4701 28.1926C23.8175 29.0828 23.8426 31.6255 23.5454 35.8205C22.7926 36.0122 22.2069 36.1408 21.7883 36.2066C21.0462 36.3232 20.2405 36.3886 19.4488 36.4139C18.6571 36.4391 18.3822 36.4363 17.2876 36.3344C16.558 36.2665 15.8577 36.1545 15.1868 35.9983Z"
                  fill="currentColor"
                />
              </svg>
            </Link>
          </motion.div>

          {/* Website Title */}
          <motion.div
            drag
            dragSnapToOrigin
            dragTransition={{ bounceStiffness: 300, bounceDamping: 15 }}
            className="cursor-pointer"
          >
            <Link href="https://ghd-snippets.vercel.app/">GHD Snippets</Link>
          </motion.div>

          {/* Sun SVG */}
          <motion.div
            drag
            dragSnapToOrigin
            dragTransition={{ bounceStiffness: 300, bounceDamping: 15 }}
            className="cursor-pointer"
          >
            <svg
              className="w-9 h-9"
              viewBox="0 0 38 38"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M19 29.2917C21.7295 29.2917 24.3473 28.2074 26.2773 26.2773C28.2074 24.3473 29.2917 21.7295 29.2917 19C29.2917 16.2705 28.2074 13.6528 26.2773 11.7227C24.3473 9.79264 21.7295 8.70834 19 8.70834C16.2705 8.70834 13.6528 9.79264 11.7227 11.7227C9.79264 13.6528 8.70834 16.2705 8.70834 19C8.70834 21.7295 9.79264 24.3473 11.7227 26.2773C13.6528 28.2074 16.2705 29.2917 19 29.2917V29.2917Z"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M30.305 30.305L30.0992 30.0992M30.0992 7.90082L30.305 7.69499L30.0992 7.90082ZM7.69499 30.305L7.90082 30.0992L7.69499 30.305ZM19 3.29332V3.16666V3.29332ZM19 34.8333V34.7067V34.8333ZM3.29332 19H3.16666H3.29332ZM34.8333 19H34.7067H34.8333ZM7.90082 7.90082L7.69499 7.69499L7.90082 7.90082Z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </motion.div>
        </div>
      </div>
    </>
  );
}
