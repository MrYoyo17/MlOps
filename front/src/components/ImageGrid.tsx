import { ArrowUp } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useInView } from 'react-intersection-observer';

import { imagesApi } from '@/api-client';
import { ImageCard } from '@/components/ImageCard';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { useInfiniteQuery } from '@tanstack/react-query';

import type { ImageDto } from '@/types/ImageDto';


interface ImageGridProps {
    filters: {
        beard?: boolean;
        mustache?: boolean;
        glasses?: boolean;
        hairColor?: 'blond' | 'lightBrown' | 'red' | 'darkBrown' | 'grayBlue';
        hairLength?: 'long' | 'short' | 'bald';
        model?: string;
    };
}

export function ImageGrid({ filters }: ImageGridProps) {
    const { ref, inView } = useInView();
    const [showBackToTop, setShowBackToTop] = useState(false);

    const {
        data,
        fetchNextPage,
        hasNextPage,
        isFetchingNextPage,
        status,
    } = useInfiniteQuery({
        queryKey: ['images', filters],
        queryFn: async ({ pageParam = 0 }) => {
            return imagesApi.imagesControllerFindAll({
                limit: 20,
                offset: pageParam,
                ...filters,
            });
        },
        initialPageParam: 0,
        getNextPageParam: (lastPage, allPages) => {
            return lastPage.length === 20 ? allPages.length * 20 : undefined;
        },
    });

    useEffect(() => {
        if (inView && hasNextPage) {
            fetchNextPage();
        }
    }, [inView, fetchNextPage, hasNextPage]);

    useEffect(() => {
        const handleScroll = () => {
            setShowBackToTop(window.scrollY > 400);
        };

        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const scrollToTop = () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    if (status === 'pending') {
        return (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {[...Array(8)].map((_, i) => (
                    <Skeleton key={i} className="aspect-square rounded-md" />
                ))}
            </div>
        );
    }

    if (status === 'error') {
        return <div className="text-center text-red-500">Error loading images</div>;
    }

    const allImages = data?.pages.flatMap((page) => page) || [];

    // Deduplicate images by id to avoid key conflicts
    const images = Array.from(
        new Map(allImages.map((image: ImageDto) => [image.id, image])).values()
    );

    if (images.length === 0) {
        return <div className="text-center text-muted-foreground py-8">No images found</div>;
    }

    return (
        <div className="space-y-4 relative">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {images.map((image: ImageDto) => (
                    <ImageCard key={image.id} image={image} />
                ))}
            </div>

            {/* Loading indicator for infinite scroll */}
            <div ref={ref} className="flex justify-center py-4">
                {isFetchingNextPage && <Skeleton className="h-8 w-8 rounded-full" />}
            </div>

            {/* Back to Top Button */}
            {showBackToTop && (
                <Button
                    onClick={scrollToTop}
                    className="fixed bottom-8 right-8 rounded-full h-12 w-12 shadow-lg z-50"
                    size="icon"
                >
                    <ArrowUp className="h-5 w-5" />
                </Button>
            )}
        </div>
    );
}