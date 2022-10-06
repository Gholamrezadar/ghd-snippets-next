import { Dispatch, SetStateAction } from 'react';
// import SyntaxHighlighter from 'react-syntax-highlighter';
// import {
//   atomOneDark,
//   atomOneLight,
// } from 'react-syntax-highlighter/dist/cjs/styles/hljs';
import ISnippet from '../lib/ISnippet';

const Snippet = ({
  snippet,
  tags,
  setCopied,
}: {
  snippet: ISnippet;
  tags: string[];
  setCopied: Dispatch<SetStateAction<boolean>>;
}) => {
  const copyToClipboard = async (content: string) => {
    // Toast
    // alert(content);
    setCopied(true);
    setTimeout(() => {
      setCopied(false);
    }, 1000);
    // Write the content to clipboard
    await navigator.clipboard.writeText(content);
  };
  const tagColors = ['bg-red-500', 'bg-blue-500', 'bg-green-500'];
  const tagList = snippet.tags.map((tag) => {
    // Index used for picking tag pill(badge) colors
    let i = tags.indexOf(tag) % tagColors.length;

    return (
      // Tag Badge
      <span
        className={
          'text-xs inline-block py-1 px-1.5 mx-0.5 leading-none text-center whitespace-nowrap align-baselinetext-white rounded-full' +
          ' ' +
          tagColors[i]
        }
        key={tag}
      >
        {tag}
      </span>
    );
  });
  return (
    <>
      <div className="w-full max-w-xl shadow-lg mb-4">
        <div className="overflow-hidden">
          <div className="dark:bg-ghd-dark-dark flex flex-row justify-between rounded-t-xl px-7 py-2">
            <div>
              <div className="text-md text-ghd-dark-text mb-1">
                {snippet.title} {tagList}
              </div>

              <div className="text-xs text-ghd-dark-muted-text">
                {snippet.subtitle}
              </div>
            </div>
            {/* Copy button */}
            <div
              className="flex justify-center items-center cursor-pointer"
              onClick={() => {
                copyToClipboard(snippet.content);
              }}
            >
              {/* Copy svg */}
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M16 12.9V17.1C16 20.6 14.6 22 11.1 22H6.9C3.4 22 2 20.6 2 17.1V12.9C2 9.4 3.4 8 6.9 8H11.1C14.6 8 16 9.4 16 12.9Z"
                  stroke="#535559"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M22 6.9V11.1C22 14.6 20.6 16 17.1 16H16V12.9C16 9.4 14.6 8 11.1 8H8V6.9C8 3.4 9.4 2 12.9 2H17.1C20.6 2 22 3.4 22 6.9Z"
                  stroke="#535559"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
          </div>
          <div className="dark:bg-gray-900 rounded-b-xl dark:text-gray-400">
            {/* <SyntaxHighlighter
              language="python"
              style={atomOneDark}
              showLineNumbers
              lineNumberStyle={{ color: '#ffffff11' }}
              customStyle={{
                fontSize: '1rem',
                scrollbarColor: '#6969dd #e0e0e0',
                borderRadius: '0 0 0.75rem 0.75rem ',
              }}
            >
              {snippet.content}
            </SyntaxHighlighter> */}
            <pre className="dark:bg-ghd-dark-code-bg p-4 pb-1 overflow-hidden rounded-b-lg">
              <div className="card-box">{snippet.content}</div>
            </pre>
          </div>
        </div>
      </div>
    </>
  );
};

export default Snippet;
