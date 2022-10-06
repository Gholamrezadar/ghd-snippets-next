const CopiedToast = ({ text, copied }: { text: string; copied: boolean }) => {
  return (
    <div
      className="fixed right-10 bottom-5 bg-green-100 rounded-lg py-5 px-6 mb-3 text-base text-green-700 inline-flex items-center transition ease-in duration-300"
      role="alert"
      style={copied ? { opacity: '1.0' } : { opacity: '0.0' }}
    >
      <svg
        aria-hidden="true"
        focusable="false"
        data-prefix="fas"
        data-icon="check-circle"
        className="w-4 h-4 mr-2 fill-current"
        role="img"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 512 512"
      >
        <path
          fill="currentColor"
          d="M504 256c0 136.967-111.033 248-248 248S8 392.967 8 256 119.033 8 256 8s248 111.033 248 248zM227.314 387.314l184-184c6.248-6.248 6.248-16.379 0-22.627l-22.627-22.627c-6.248-6.249-16.379-6.249-22.628 0L216 308.118l-70.059-70.059c-6.248-6.248-16.379-6.248-22.628 0l-22.627 22.627c-6.248 6.248-6.248 16.379 0 22.627l104 104c6.249 6.249 16.379 6.249 22.628.001z"
        ></path>
      </svg>
      {text}
    </div>
  );
};

export default CopiedToast;
