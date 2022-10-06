import create from 'zustand';

export interface AppState {
  filter: string;
  setFilter: (filter: string) => void;

  tags: string[];
  fetchTags: any;
  setTags: (tags: string[]) => void;

  selectedTags: string[];
  setSelectedTags: (tags: string[]) => void;

  copied: boolean;
  setCopied: (copied: boolean) => void;
}

const useStore = create<AppState>((set) => ({
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

  // Is 'A' snippet copied
  copied: false,
  setCopied: (copied) =>
    set((state) => ({
      ...state,
      copied,
    })),
}));

export default useStore;
