// import { AnimatePresence } from 'framer-motion';
import { AnimatePresence } from 'framer-motion';
import { useEffect, useState } from 'react';
import ISnippet from '../lib/ISnippet';
import useStore from '../lib/store';
import Snippet from './Snippet';

const SnippetsContainer = ({ snippets }: { snippets: ISnippet[] }) => {
  // Zustand store
  const setCopied = useStore((state) => state.setCopied);
  const filter = useStore((state) => state.filter);
  const selectedTags = useStore((state) => state.selectedTags);
  const setNumFilteredSnippets = useStore(
    (state) => state.setNumFilteredSnippets
  );
  const [snippetsListState, setSnippetsListState] = useState([]);

  useEffect(() => {
    let snippetsList;
    // If no tags are selected don't do filtering based on tags(include results from every tag)
    if (selectedTags.length === 0) {
      snippetsList = snippets
        .filter(({ title }) =>
          title.toLowerCase().includes(filter.toLowerCase())
        )
        .map((snippet) => {
          return (
            <Snippet snippet={snippet} key={snippet.id} setCopied={setCopied} />
          );
        });
    }
    // Filtering based on tags then based on filter text
    else {
      snippetsList = snippets
        .filter(({ tags }) => selectedTags.some((item) => tags.includes(item)))
        .filter(({ title }) =>
          title.toLowerCase().includes(filter.toLowerCase())
        )
        .map((snippet) => {
          return (
            <Snippet snippet={snippet} key={snippet.id} setCopied={setCopied} />
          );
        });
    }
    // Set current snippets as a state
    setSnippetsListState(snippetsList);
    // Set the number of current snippets
    setNumFilteredSnippets(snippetsList.length);
  }, [filter, selectedTags]);

  return (
    <>
      <div className="flex items-center w-full flex-col px-2">
        <AnimatePresence>
        {snippetsListState}
        </AnimatePresence>
      </div>
    </>
  );
};

export default SnippetsContainer;
