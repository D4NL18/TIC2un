import { Component, ViewChild } from '@angular/core';
import { KcmeansResultsComponent } from '../../components/kcmeans-results/kcmeans-results.component';
import { ButtonComponent } from '../../components/button/button.component';
import { NavComponent } from '../../components/nav/nav.component';

@Component({
  selector: 'app-kcmeans',
  standalone: true,
  imports: [KcmeansResultsComponent, ButtonComponent, NavComponent],
  templateUrl: './kcmeans.component.html',
  styleUrl: './kcmeans.component.scss'
})
export class KcmeansComponent {
  @ViewChild(KcmeansResultsComponent) kcResultsComponent!: KcmeansResultsComponent

  onTrainKC() {
    if(this.kcResultsComponent) {
      this.kcResultsComponent.trainKC()
    }
  }
}
