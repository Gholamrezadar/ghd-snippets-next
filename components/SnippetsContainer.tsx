import { Dispatch, SetStateAction } from 'react';
import ISnippet from '../lib/ISnippet';
import useStore from '../lib/store';
import Snippet from './Snippet';

const SnippetsContainer = ({
  snippets,
  setCopied,
}: {
  snippets: ISnippet[];
  setCopied: Dispatch<SetStateAction<boolean>>;
}) => {
  // Zustand store
  const filter = useStore((state) => state.filter);
  const selectedTags = useStore((state) => state.selectedTags);

  let tags = new Set(
    snippets.flatMap((snippet) => {
      return snippet.tags;
    })
  );

  let snippetsList;
  // If no tags are selected don't do filtering based on tags(include results from every tag)
  if (selectedTags.length === 0) {
    snippetsList = snippets
      .filter(({ title }) => title.toLowerCase().includes(filter.toLowerCase()))
      .map((snippet) => {
        return (
          <Snippet
            snippet={snippet}
            tags={[...tags]}
            key={snippet.id}
            setCopied={setCopied}
          />
        );
      });
  }
  // Filtering based on tags then based on filter text
  else {
    snippetsList = snippets
      .filter(({ tags }) => selectedTags.some((item) => tags.includes(item)))
      .filter(({ title }) => title.toLowerCase().includes(filter.toLowerCase()))
      .map((snippet) => {
        return (
          <Snippet
            snippet={snippet}
            tags={[...tags]}
            key={snippet.id}
            setCopied={setCopied}
          />
        );
      });
  }

  return (
    <>
      <div className="flex items-center w-full flex-col">{snippetsList}</div>
    </>
  );
};

export default SnippetsContainer;
