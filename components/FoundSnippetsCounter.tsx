import useStore from '../lib/store';

const foundSnippetsCounter = () => {
  const numFilteredSnippets = useStore((state) => state.numFilteredSnippets);

  return (
    <div className="mt-5 text-ghd-dark-muted-text">
      Found {numFilteredSnippets} Snippets
    </div>
  );
};

export default foundSnippetsCounter;
