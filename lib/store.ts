import create from 'zustand';

export interface AppState {
  filter: string;
  setFilter: (filter: string) => void;

  tags: string[];
  fetchTags: any;
  setTags: (tags: string[]) => void;

  selectedTags: string[];
  setSelectedTags: (tags: string[]) => void;
  updateSelectedTags: (tag: string, checked: boolean) => void;

  copied: boolean;
  setCopied: (copied: boolean) => void;

  numFilteredSnippets: number;
  setNumFilteredSnippets: (number) => void;
}

const useStore = create<AppState>((set, get) => ({
  // Filter typed in the searchbar
  filter: '',
  setFilter: (filter) =>
    set((state) => ({
      ...state,
      filter,
    })),

  // All available Tags
  tags: [],
  fetchTags: async () => {
    const data = (await import('../lib/data')).tags;
    set({ tags: data });
  },
  setTags: (tags) =>
    set((state) => ({
      ...state,
      tags,
    })),

  // Selected Tags
  selectedTags: [],
  setSelectedTags: (selectedTags) =>
    set((state) => ({
      ...state,
      selectedTags,
    })),
  // Update selected tags
  updateSelectedTags: (tag, checked) => {
    if (checked) get().setSelectedTags([...get().selectedTags, tag]);
    else
      get().setSelectedTags(
        get().selectedTags.filter((value) => value !== tag)
      );
  },

  // Is 'A' snippet copied
  copied: false,
  setCopied: (copied) =>
    set((state) => ({
      ...state,
      copied,
    })),

  numFilteredSnippets: 0,
  setNumFilteredSnippets: (numFilteredSnippets) =>
    set((state) => ({ ...state, numFilteredSnippets })),
}));

export default useStore;
