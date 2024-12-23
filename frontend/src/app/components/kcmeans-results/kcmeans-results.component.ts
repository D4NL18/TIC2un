import { Component, EventEmitter, Output } from '@angular/core';
import { KService } from '../../services/k-means.service';
import { CService } from '../../services/c-means.service';
import { LoadingSpinnerComponent } from '../loading-spinner/loading-spinner.component';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-kcmeans-results',
  standalone: true,
  imports: [LoadingSpinnerComponent, CommonModule],
  templateUrl: './kcmeans-results.component.html',
  styleUrls: ['./kcmeans-results.component.scss']
})
export class KcmeansResultsComponent {
  @Output() trainRequested = new EventEmitter<void>()

  imageUrlK: string | undefined
  imageUrlC: string | undefined
  isLoading: boolean = false;

  constructor(private kService: KService, private cService: CService) { }

  trainKC() {

    this.isLoading = true;

    this.kService.trainK().subscribe({
      next: (response) => {
        console.log("Treinamento K-means Concluído", response);
        this.fetchImageK();
      },
      error: (err) => {
        console.log("Erro ao treinar K-means:", err);
        this.isLoading = false;
      },
      complete: () => {
        this.isLoading = false;
      }
    });
    this.cService.trainC().subscribe({
      next: (response) => {
        console.log("Treinamento C-means Concluído", response);
        this.fetchImageC();
      },
      error: (err) => {
        console.log("Erro ao treinar C-means:", err);
        this.isLoading = false;
      },
      complete: () => {
        this.isLoading = false;
      }
    });
  }

  fetchImageK() {
    this.kService.getImage().subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob);
        this.imageUrlK = url;
      },
      error: (err) => {
        console.error('Erro ao buscar imagem K:', err);
      }
    });
  }

  fetchImageC() {
    this.cService.getImage().subscribe({
      next: (blob: Blob | MediaSource) => {
        const url = URL.createObjectURL(blob);
        this.imageUrlC = url;
      },
      error: (err: any) => {
        console.error('Erro ao buscar imagem C:', err);
      }
    });
  }
}
